#!/usr/bin/env python3
"""User-run, held-out positive/negative benchmark. Never promotes models."""

import argparse
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from celebrity_verifier import configured_verifier, jpeg_bytes
from utils import get_actor_folder_name, get_average_embedding_path, load_pickle
from utils_deepface import get_face_embeddings
from validation import classify_candidate, file_hash, unit, validate_model_metadata, write_json


def run_benchmark(manifest, output, use_aws=False):
    spec = importlib.util.spec_from_file_location('headshots', Path(__file__).with_name('33_extract_video_headshots.py'))
    headshots = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(headshots)
    manifest = Path(manifest)
    cases = json.loads(manifest.read_text(encoding='utf-8'))
    if not isinstance(cases, list) or not cases:
        raise ValueError('Manifest must be a nonempty list of labeled cases')
    # Enforce both labels for every tested actor and prevent training leakage.
    groups = {}
    training_hashes = set()
    for metadata in Path('04_models').glob('*.quality.json'):
        training_hashes.update(json.loads(metadata.read_text(encoding='utf-8')).get('training_images', {}).values())
    seen = set()
    for case in cases:
        if type(case.get('expected_match')) is not bool:
            raise ValueError('Every case needs an explicit boolean expected_match')
        groups.setdefault(case['actor'], set()).add(case['expected_match'])
        path = manifest.parent / case['image']
        digest = file_hash(path)
        if digest in training_hashes:
            raise ValueError(f"Training image leaked into benchmark: {case['image']}")
        if (case['actor'], digest) in seen:
            raise ValueError('Duplicate benchmark image for actor')
        seen.add((case['actor'], digest))
    if any(labels != {True, False} for labels in groups.values()):
        raise ValueError('Every actor needs held-out positives AND negatives')
    verifier, mapping = configured_verifier(Path(output).with_suffix('.aws.sqlite')) if use_aws else (None, {})
    if use_aws and verifier is None:
        raise ValueError('Set CELEBRITY_VERIFIER=aws_required before --aws')
    threshold = float(os.getenv('OPERATIONS_HEADSHOT_MATCH_THRESHOLD', '0.4'))
    margin = float(os.getenv('OPERATIONS_MIN_MATCH_MARGIN', '0.08'))
    counts, rows = Counter(), []
    for case in cases:
        path = manifest.parent / case['image']
        model = get_average_embedding_path(case['actor'], 'models')
        if validate_model_metadata(model) is None:
            raise ValueError(f"Unvalidated model: {case['actor']}")
        reference = unit(load_pickle(model))
        competitors = headshots.load_competitors(model, reference)
        faces = get_face_embeddings(path)
        if faces is None:
            raise ValueError(f"Embedding extraction failed: {case['image']}")
        decision = dict(status='unknown_or_multiple_faces')
        if len(faces) == 1:
            decision = classify_candidate(faces[0]['embedding'], reference, competitors, threshold, margin)
        local_accepted = decision['status'] == 'accepted'
        verdict = dict(status='not_requested')
        # Benchmark AWS independently, including local rejections; it never
        # overrides a local rejection when calculating the combined result.
        if verifier and len(faces) == 1:
            verdict = verifier.verify(jpeg_bytes(path, faces[0]['bounding_box']),
                                      mapping.get(get_actor_folder_name(case['actor'])), case['actor'])
        elif verifier:
            verdict = dict(status='unknown_or_multiple_faces')
        accepted = local_accepted and (not verifier or verdict['status'] == 'verified')
        key = ('true_positive' if accepted else 'false_negative') if case['expected_match'] else ('false_positive' if accepted else 'true_negative')
        counts[key] += 1
        counts['local_accepted'] += int(local_accepted)
        if verifier and verdict['status'] not in ('verified', 'identity_mismatch'):
            counts['cloud_abstentions'] += 1
        rows.append(dict(**case, decision=decision, cloud=verdict, accepted=accepted, result=key))
    report = dict(cases=rows, counts=dict(counts), threshold=threshold, margin=margin,
                  scope='single-photo gate; temporal corroboration is tested separately',
                  note='Labels are supplied by the manifest author, not inferred by this benchmark')
    write_json(output, report)
    print(json.dumps(report['counts']))
    return report


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('--output', default='benchmark-results.json')
    parser.add_argument('--aws', action='store_true', help='Send benchmark images to AWS; consumes API requests')
    args = parser.parse_args()
    run_benchmark(args.manifest, args.output, args.aws)


if __name__ == '__main__':
    main()
