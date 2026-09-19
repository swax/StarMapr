"""Conservative, testable gates. Cohesion is not proof of a person's identity."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


def embedding_spec():
    try:
        deepface_version = version('deepface')
    except PackageNotFoundError:
        deepface_version = 'unavailable'
    return dict(schema=1, model='ArcFace', detector='opencv', normalization='base',
                align=True, deepface_version=deepface_version)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    """Replace a report only after the full new report has been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError('Embedding must be a finite, nonempty vector')
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError('Embedding has zero length')
    return vector / norm


def similarity(left, right):
    return float(np.clip(np.dot(unit(left), unit(right)), -1, 1))


@dataclass(frozen=True)
class QualityPolicy:
    min_images: int = 15
    min_median: float = 0.55
    min_p10: float = 0.45

    def __post_init__(self):
        if self.min_images < 3 or not (0 <= self.min_p10 <= self.min_median <= 1):
            raise ValueError('Invalid model quality policy')

    @classmethod
    def from_env(cls, min_images=None):
        return cls(int(min_images if min_images is not None else os.getenv('TRAINING_MIN_IMAGES', '15')),
                   float(os.getenv('TRAINING_MIN_MEDIAN_COHESION', '0.55')),
                   float(os.getenv('TRAINING_MIN_P10_COHESION', '0.45')))


def assess_reference(embeddings, policy=None):
    """Leave each sample out of its centroid, so it cannot boost its own score."""
    policy = policy or QualityPolicy.from_env()
    report = dict(accepted=False, count=len(embeddings), policy=asdict(policy),
                  reason='insufficient_samples')
    if len(embeddings) < policy.min_images:
        return None, report
    try:
        rows = np.stack([unit(row) for row in embeddings])
        total = rows.sum(axis=0)
        scores = [similarity(row, total - row) for row in rows]
        centroid = unit(total)
    except (ValueError, TypeError):
        report['reason'] = 'invalid_embeddings'
        return None, report
    median, p10 = float(np.median(scores)), float(np.quantile(scores, 0.1))
    accepted = median >= policy.min_median and p10 >= policy.min_p10
    report.update(accepted=accepted, reason='cohesive' if accepted else 'incoherent_references',
                  median=median, p10=p10)
    return centroid if accepted else None, report


def metadata_path(model_path):
    return Path(model_path).with_suffix('.quality.json')


def validate_model_metadata(model_path, policy=None):
    """Legacy or changed models remain intact, but do not produce new headshots."""
    try:
        report = json.loads(metadata_path(model_path).read_text(encoding='utf-8'))
        quality = report['quality']
        policy = policy or QualityPolicy.from_env()
        if (report['model_sha256'] != file_hash(model_path)
                or report['embedding_spec'] != embedding_spec()
                or quality.get('accepted') is not True
                or not np.isfinite([quality['count'], quality['median'], quality['p10']]).all()
                or quality['count'] < policy.min_images
                or quality['median'] < policy.min_median
                or quality['p10'] < policy.min_p10):
            return None
        return report
    except (OSError, ValueError, KeyError, TypeError):
        return None


def classify_candidate(embedding, reference, competitors, threshold=0.4, margin=0.08):
    if not (0 <= threshold <= 1 and 0 < margin <= 1):
        raise ValueError('Invalid matching threshold or margin')
    score = similarity(embedding, reference)
    rivals = {name: similarity(embedding, other) for name, other in competitors.items()}
    rival = max(rivals, key=rivals.get) if rivals else None
    rival_score = rivals[rival] if rival else None
    reason = 'accepted'
    if score < threshold:
        reason = 'below_threshold'
    elif rival_score is not None and score - rival_score < margin:
        reason = 'ambiguous'
    return dict(status=reason, score=score, competitor=rival, competitor_score=rival_score,
                margin=None if rival_score is None else score - rival_score)


def corroborated_candidates(candidates, min_frames=2, min_frame_gap=25, min_similarity=0.6):
    """Require the *same* candidate appearance in separated frames; no identity claim."""
    if min_frames < 2 or min_frame_gap < 1 or not 0 <= min_similarity <= 1:
        raise ValueError('Invalid corroboration policy')
    result = []
    for candidate in sorted(candidates, key=lambda c: c['decision']['score'], reverse=True):
        supporters = [candidate['frame_position']]
        for other in sorted(candidates, key=lambda c: c['frame_position']):
            position = other['frame_position']
            if any(abs(position - previous) < min_frame_gap for previous in supporters):
                continue
            if similarity(candidate['face']['embedding'], other['face']['embedding']) >= min_similarity:
                supporters.append(position)
        if len(supporters) >= min_frames:
            result.append(candidate)
    return result
