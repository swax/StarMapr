#!/usr/bin/env python3
"""Extract corroborated, unambiguous headshots or record an explicit abstention."""

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
from dotenv import load_dotenv

from celebrity_verifier import configured_verifier, jpeg_bytes
from utils import (get_actor_folder_name, get_average_embedding_path, get_env_float,
                   get_headshot_crop_coordinates, load_pickle, print_error)
from utils_deepface import cache_spec
from validation import (classify_candidate, corroborated_candidates, unit,
                        validate_model_metadata, write_json)

load_dotenv()


def load_competitors(model_path, reference):
    competitors = {}
    for path in sorted(model_path.parent.glob('*_average_embedding.pkl')):
        if path == model_path:
            continue
        vector = load_pickle(path)
        if vector is None:
            raise ValueError(f'Unreadable competitor model: {path.name}')
        vector = unit(vector)
        if vector.shape != reference.shape:
            raise ValueError(f'Incompatible competitor model: {path.name}')
        # Even a legacy rival can veto an ambiguous match; it cannot authorize one.
        competitors[path.stem.removesuffix('_average_embedding')] = vector
    return competitors


def scan_candidates(frames_dir, reference, competitors, threshold, margin):
    candidates, rejected = [], Counter()
    files = sorted(frames_dir.glob('*.pkl'))
    if not files:
        raise ValueError('No processed frames; run 32_extract_frame_faces.py first')
    for path in files:
        data = load_pickle(path)
        if not isinstance(data, dict):
            raise ValueError(f'Unreadable frame data: {path.name}')
        frame_file = data.get('frame_file', path.stem + '.jpg')
        if Path(frame_file).name != frame_file or not Path(frame_file).stem.isdigit():
            raise ValueError(f'Invalid frame filename: {frame_file}')
        frame = frames_dir / frame_file
        if data.get('cache_spec') != cache_spec(frame):
            raise ValueError(f'Stale cache for {frame_file}; run 32_extract_frame_faces.py')
        for face in data.get('faces', []):
            if not face.get('isHeadshotable', False):
                rejected['not_headshotable'] += 1
                continue
            decision = classify_candidate(face['embedding'], reference, competitors, threshold, margin)
            if decision['status'] != 'accepted':
                rejected[decision['status']] += 1
                continue
            candidates.append(dict(frame_file=frame_file, frame_position=int(Path(frame_file).stem),
                                   face=face, decision=decision))
    return candidates, rejected


def crop_candidate(frames_dir, candidate):
    image = cv2.imread(str(frames_dir / candidate['frame_file']))
    if image is None:
        raise ValueError(f"Cannot read frame {candidate['frame_file']}")
    coords = get_headshot_crop_coordinates(candidate['face']['bounding_box'], image.shape[1], image.shape[0])
    if coords['clipped']:
        return None
    crop = image[coords['y_start']:coords['y_end'], coords['x_start']:coords['x_end']]
    success, encoded = cv2.imencode('.jpg', crop)
    if not success:
        raise ValueError('Could not encode headshot')
    return encoded.tobytes()


def publish_result(folder, images, report):
    """Reports name the exact current output; old loose JPGs never count as success."""
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder.parent) as temp:
        stage = Path(temp)
        for name, data in images.items():
            (stage / name).write_bytes(data)
        # Limit cleanup to files owned by this headshot stage.
        for old in folder.iterdir():
            if old.is_file() and old.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                old.unlink()
        for image in stage.iterdir():
            shutil.move(str(image), folder / image.name)
        write_json(folder / 'result.json', report)


def extract_top_headshots(actor_name, video_folder_path, threshold=0.4, dry_run=False):
    video = Path(video_folder_path)
    frames = video / 'frames'
    if not frames.is_dir():
        raise FileNotFoundError(f'Frames directory not found: {frames}')
    actor = get_actor_folder_name(actor_name)
    if not actor:
        raise ValueError('Actor name must contain letters')
    output = video / 'headshots' / actor
    model_path = get_average_embedding_path(actor_name, 'models')
    margin = float(os.getenv('OPERATIONS_MIN_MATCH_MARGIN', '0.08'))
    gap = int(os.getenv('OPERATIONS_MIN_FRAME_GAP', '25'))
    min_frames = int(os.getenv('OPERATIONS_MIN_CONFIRMING_FRAMES', '2'))
    support_threshold = float(os.getenv('OPERATIONS_CORROBORATION_THRESHOLD', '0.6'))
    report = dict(schema=1, actor=actor_name, status='model_unvalidated', headshots=[],
                  retryable=False, threshold=threshold, min_margin=margin,
                  min_confirming_frames=min_frames, min_frame_gap=gap,
                  corroboration_threshold=support_threshold, dry_run=dry_run)
    images = {}
    model_report = validate_model_metadata(model_path)
    if model_report:
        reference = unit(load_pickle(model_path))
        competitors = load_competitors(model_path, reference)
        report['competitor_models'] = list(competitors)
        report['model_sha256'] = model_report['model_sha256']
        candidates, rejected = scan_candidates(frames, reference, competitors, threshold, margin)
        supported = corroborated_candidates(candidates, min_frames, gap, support_threshold)
        rejected['insufficient_corroboration'] += len(candidates) - len(supported)
        verifier, identity_map = configured_verifier(video / 'celebrity-cache.sqlite',
            json.loads(os.getenv('STARMAPR_VIDEO_ACTORS', json.dumps([actor_name]))))
        report['verification'] = 'aws_required' if verifier else 'local_only'
        report['status'] = 'no_reliable_headshot'
        report['retryable'] = True
        selected_positions = []
        attempted = 0
        for candidate in supported:
            if len(images) >= 5 or attempted >= 10:
                break
            if any(abs(candidate['frame_position'] - p) < gap for p in selected_positions):
                continue
            attempted += 1
            data = crop_candidate(frames, candidate)
            if data is None:
                rejected['clipped_crop'] += 1
                continue
            if verifier:
                # A dry run must not call AWS or create a cache/budget database.
                verdict = dict(status='not_run_dry_run') if dry_run else verifier.verify(
                    jpeg_bytes(frames / candidate['frame_file'], candidate['face']['bounding_box']),
                    identity_map.get(actor), actor_name)
            else:
                verdict = dict(status='disabled')
            if verifier and verdict['status'] != 'verified':
                rejected[verdict['status']] += 1
                if verdict['status'] in ('missing_identity_mapping', 'budget_exhausted', 'actor_budget_exhausted',
                                         'verifier_unavailable', 'not_run_dry_run'):
                    report.update(status=verdict['status'], retryable=False)
                    break
                continue
            name = (f"{actor}_match_{candidate['decision']['score']:.3f}_"
                    f"position_{Path(candidate['frame_file']).stem}.jpg")
            images[name] = data
            selected_positions.append(candidate['frame_position'])
            report['headshots'].append(dict(file=name, frame=candidate['frame_file'],
                                            face_id=candidate['face'].get('face_id'),
                                            decision=candidate['decision'], verification=verdict))
        report['rejections'] = dict(rejected)
        if images:
            report.update(status='accepted', retryable=False)
    if not dry_run:
        publish_result(output, images, report)
    print(json.dumps(report, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('actor_name')
    parser.add_argument('video_folder_path')
    parser.add_argument('--threshold', '-t', type=float,
                        default=get_env_float('OPERATIONS_HEADSHOT_MATCH_THRESHOLD', 0.4))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        extract_top_headshots(args.actor_name, args.video_folder_path, args.threshold, args.dry_run)
    except Exception as exc:
        print_error(str(exc))
        sys.exit(1)


if __name__ == '__main__':
    main()
