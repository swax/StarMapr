"""Optional AWS adapter. No credentials, images, or API responses are logged.

Only an exact catalog name or configured celebrity ID can pass. Fuzzy names, service failures,
unknown faces and multiple faces abstain. SDK imports and calls are lazy.
"""

import hashlib
import json
import os
import sqlite3
import unicodedata
from contextlib import closing
from pathlib import Path


def jpeg_bytes(path, bounding_box=None):
    """Normalize supported local image formats for the AWS JPEG/PNG API."""
    import cv2
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f'Unreadable image: {Path(path).name}')
    if bounding_box:
        # Verify the detected face, not another recognizable face in a padded crop.
        x, y, w, h = (int(bounding_box[key]) for key in ('x', 'y', 'w', 'h'))
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            raise ValueError('Invalid face bounding box')
        image = image[y:y+h, x:x+w]
    ok, encoded = cv2.imencode('.jpg', image)
    if not ok:
        raise ValueError('Could not encode image')
    return encoded.tobytes()


def normalized_name(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def evaluate_response(response, expected_id, confidence, expected_name=None):
    faces = response.get('CelebrityFaces', [])
    if len(faces) != 1 or response.get('UnrecognizedFaces'):
        return dict(status='unknown_or_multiple_faces')
    face = faces[0]
    if not face.get('Id'):
        return dict(status='unknown_identity')
    if expected_id:
        identity_matches = face['Id'] == expected_id
    else:
        identity_matches = bool(expected_name and face.get('Name') and
                                normalized_name(face['Name']) == normalized_name(expected_name))
    if not identity_matches:
        return dict(status='identity_mismatch')
    score = float(face.get('MatchConfidence', 0))
    if not confidence <= score <= 100:
        return dict(status='low_confidence')
    return dict(status='verified', celebrity_id=face['Id'], confidence=score,
                matched_by='id' if expected_id else 'exact_catalog_name')


class CelebrityVerifier:
    def __init__(self, database, *, region='us-east-1', max_requests=40,
                 confidence=99.0, client=None, actor_limits=None):
        if max_requests < 0 or not 0 < confidence <= 100:
            raise ValueError('Invalid celebrity verifier policy')
        self.database = Path(database)
        self.region = region
        self.max_requests = max_requests
        self.confidence = confidence
        self.client = client
        self.actor_limits = actor_limits

    def _client(self):
        if self.client is None:
            import boto3
            from botocore.config import Config
            self.client = boto3.client('rekognition', region_name=self.region,
                                      config=Config(connect_timeout=5, read_timeout=15,
                                                    retries={'total_max_attempts': 1}))
        return self.client

    def verify(self, image_bytes, expected_id=None, expected_name=None):
        if not expected_id and not expected_name:
            return dict(status='missing_identity_mapping')
        if not image_bytes or len(image_bytes) > 5 * 1024 * 1024:
            return dict(status='invalid_image_size')
        key = hashlib.sha256(self.region.encode() + b'\0v2\0' + image_bytes).hexdigest()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        # A durable transaction shares the budget across actors, retries, and
        # processes. Reserve before the request; even an interrupted call counts.
        with closing(sqlite3.connect(self.database, timeout=10)) as db:
            db.execute('CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY CHECK(id=1), used INTEGER NOT NULL)')
            db.execute('INSERT OR IGNORE INTO budget VALUES (1, 0)')
            db.execute('CREATE TABLE IF NOT EXISTS actor_budget (actor TEXT PRIMARY KEY, used INTEGER NOT NULL)')
            db.commit()
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT value FROM responses WHERE key=?', (key,)).fetchone()
            if row:
                return evaluate_response(json.loads(row[0]), expected_id, self.confidence, expected_name)
            used = db.execute('SELECT used FROM budget WHERE id=1').fetchone()[0]
            if used >= self.max_requests:
                return dict(status='budget_exhausted')
            actor = normalized_name(expected_name or expected_id)
            if self.actor_limits is not None:
                limit = self.actor_limits.get(actor, 0)
                row = db.execute('SELECT used FROM actor_budget WHERE actor=?', (actor,)).fetchone()
                if (row[0] if row else 0) >= limit:
                    return dict(status='actor_budget_exhausted')
            db.execute('INSERT INTO actor_budget VALUES (?, 1) ON CONFLICT(actor) DO UPDATE SET used=used+1', (actor,))
            db.execute('UPDATE budget SET used=used+1 WHERE id=1')
            db.commit()
        try:
            response = self._client().recognize_celebrities(Image={'Bytes': image_bytes})
            # Retain only what the gate needs, not response metadata or images.
            compact = dict(CelebrityFaces=[{k: f.get(k) for k in ('Id', 'Name', 'MatchConfidence')}
                                           for f in response.get('CelebrityFaces', [])],
                           UnrecognizedFaces=[{} for _ in response.get('UnrecognizedFaces', [])])
            result = evaluate_response(compact, expected_id, self.confidence, expected_name)
        except Exception as exc:
            # Do not cache transient errors. Count them against the request cap.
            return dict(status='verifier_unavailable', error_type=type(exc).__name__)
        with closing(sqlite3.connect(self.database, timeout=10)) as db:
            db.execute('INSERT OR REPLACE INTO responses VALUES (?, ?)', (key, json.dumps(compact)))
            db.commit()
        return result


def configured_verifier(database, actor_names=None):
    mode = os.getenv('CELEBRITY_VERIFIER', 'off')
    if mode == 'off':
        return None, {}
    if mode != 'aws_required':
        raise ValueError('CELEBRITY_VERIFIER must be off or aws_required')
    mapping_path = os.getenv('AWS_CELEBRITY_ID_MAP')
    mapping = json.loads(Path(mapping_path).read_text(encoding='utf-8')) if mapping_path else {}
    if not isinstance(mapping, dict) or any(not isinstance(v, str) or not v for v in mapping.values()):
        raise ValueError('AWS_CELEBRITY_ID_MAP must contain actor slugs mapped to exact ID strings')
    budget = int(os.getenv('AWS_CELEBRITY_MAX_REQUESTS', '40'))
    limits = None
    if actor_names is not None:
        actors = sorted(set(normalized_name(name) for name in actor_names))
        # Deterministic shares persist across sampling passes and process restarts.
        count, extra = divmod(budget, len(actors) or 1)
        limits = {actor: count + (index < extra) for index, actor in enumerate(actors)}
    return CelebrityVerifier(database, region=os.getenv('AWS_REGION', 'us-east-1'),
                             max_requests=budget, actor_limits=limits,
                             confidence=float(os.getenv('AWS_CELEBRITY_MIN_CONFIDENCE', '99'))), mapping
