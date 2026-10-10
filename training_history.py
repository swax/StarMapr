"""Durable local cooldowns and accepted-image sources, independent of training folders."""

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from utils import get_actor_folder_name

DATABASE = Path(__file__).resolve().parent / '07_training_stats' / 'training.sqlite'
COOLDOWN_SECONDS = 7 * 24 * 60 * 60


def utc_text(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


class TrainingHistory:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else DATABASE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS actors (
                    actor_key TEXT PRIMARY KEY, actor_name TEXT NOT NULL,
                    cooldown_until REAL, failure_phase TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, actor_key TEXT NOT NULL,
                    kind TEXT NOT NULL, occurred_at REAL NOT NULL,
                    details TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sources (
                    event_id INTEGER NOT NULL, actor_key TEXT NOT NULL,
                    filename TEXT NOT NULL, sha256 TEXT NOT NULL,
                    page_domain TEXT, image_domain TEXT, provenance TEXT NOT NULL,
                    PRIMARY KEY (event_id, filename)
                );
                CREATE INDEX IF NOT EXISTS events_actor ON events(actor_key);
            ''')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def actor(db, name):
        key = get_actor_folder_name(name)
        db.execute('INSERT INTO actors(actor_key, actor_name) VALUES (?, ?) '
                   'ON CONFLICT(actor_key) DO UPDATE SET actor_name=excluded.actor_name', (key, name))
        return key

    @staticmethod
    def event(db, key, kind, now, **details):
        return db.execute('INSERT INTO events(actor_key, kind, occurred_at, details) VALUES (?, ?, ?, ?)',
                          (key, kind, now, json.dumps(details))).lastrowid

    def skip_if_cooling_down(self, actor_name, *, now=None):
        now = time.time() if now is None else now
        with self.connection() as db:
            # The check and increment are one transaction, including concurrent CLI jobs.
            db.execute('BEGIN IMMEDIATE')
            key = get_actor_folder_name(actor_name)
            row = db.execute('SELECT * FROM actors WHERE actor_key=?', (key,)).fetchone()
            if row is None or row['cooldown_until'] is None or now >= row['cooldown_until']:
                return None
            self.event(db, key, 'cooldown_skipped', now, cooldown_until=row['cooldown_until'])
            count = db.execute("SELECT COUNT(*) FROM events WHERE actor_key=? AND kind='cooldown_skipped'",
                               (key,)).fetchone()[0]
            return dict(retry_after=utc_text(row['cooldown_until']),
                        failure_phase=row['failure_phase'], cooldown_skips=count)

    def record_failure(self, actor_name, phase, *, now=None):
        now = time.time() if now is None else now
        until = now + COOLDOWN_SECONDS
        with self.connection() as db:
            key = self.actor(db, actor_name)
            db.execute('UPDATE actors SET cooldown_until=?, failure_phase=? WHERE actor_key=?',
                       (until, phase, key))
            self.event(db, key, 'failed', now, phase=phase, cooldown_until=until)
        return utc_text(until)

    def record_success(self, actor_name, show_name, report, *, now=None):
        now = time.time() if now is None else now
        with self.connection() as db:
            key = self.actor(db, actor_name)
            event = self.event(db, key, 'succeeded', now, show=show_name,
                               model_sha256=report['model_sha256'])
            sources = {source['filename']: source for source in report.get('training_sources', [])}
            # Only the images in the promoted model's manifest can contribute domains.
            for filename, digest in report.get('training_images', {}).items():
                source = sources.get(filename, {})
                if source.get('sha256') != digest:
                    source = {}
                db.execute('INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?)',
                           (event, key, filename, digest, source.get('page_domain'),
                            source.get('image_domain'), json.dumps(source)))
            db.execute('UPDATE actors SET cooldown_until=NULL, failure_phase=NULL WHERE actor_key=?', (key,))

    def summary(self, *, now=None):
        now = time.time() if now is None else now
        with self.connection() as db:
            db.execute('BEGIN')
            counts = dict(db.execute('SELECT kind, COUNT(*) FROM events GROUP BY kind').fetchall())
            actors = [dict(row) for row in db.execute('''
                SELECT a.actor_name, a.cooldown_until, a.failure_phase,
                    SUM(e.kind='failed') AS failed_runs,
                    SUM(e.kind='cooldown_skipped') AS cooldown_skips,
                    SUM(e.kind='succeeded') AS successful_runs
                FROM actors a JOIN events e USING(actor_key)
                GROUP BY a.actor_key ORDER BY cooldown_skips DESC, a.actor_name
            ''')]
            for actor in actors:
                until = actor['cooldown_until']
                actor['cooling_down'] = until is not None and until > now
                actor['retry_after'] = utc_text(until) if until is not None else None
                del actor['cooldown_until']
            domains = {}
            for column in ('page_domain', 'image_domain'):
                domains[column] = [dict(row) for row in db.execute(f'''
                    SELECT {column} AS domain, COUNT(DISTINCT sha256) AS images,
                           COUNT(DISTINCT actor_key) AS actors,
                           COUNT(DISTINCT event_id) AS successful_runs
                    FROM sources WHERE {column} IS NOT NULL
                    GROUP BY {column} ORDER BY images DESC, actors DESC, domain
                ''')]
            unknown = db.execute('SELECT COUNT(*) FROM sources WHERE page_domain IS NULL').fetchone()[0]
            return dict(failed_runs=counts.get('failed', 0), cooldown_skips=counts.get('cooldown_skipped', 0),
                        successful_runs=counts.get('succeeded', 0),
                        active_cooldowns=sum(actor['cooling_down'] for actor in actors),
                        accepted_image_uses_without_page_domain=unknown, actors=actors, **domains)
