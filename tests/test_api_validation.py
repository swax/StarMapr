"""API/CLI integration checks use fake artifacts and mocked processing only."""

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def script(filename):
    spec = importlib.util.spec_from_file_location(filename.removesuffix('.py'), ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ApiValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / '05_videos' / 'synthetic'
        self.actor_dir = self.video / 'headshots' / 'example'
        self.actor_dir.mkdir(parents=True)
        (self.actor_dir / 'accepted.jpg').write_bytes(b'synthetic')
        (self.actor_dir / 'stale.jpg').write_bytes(b'synthetic')
        self.api = script('90_run_api_server.py')
        for field, value in [('ROOT_DIR', self.root), ('VIDEOS_DIR', self.root / '05_videos')]:
            change = patch.object(self.api, field, value)
            change.start()
            self.addCleanup(change.stop)

    def test_only_current_accepted_manifest_files_are_exposed(self):
        outcomes = {'Example': {'status': 'accepted', 'headshots': [
            {'file': 'accepted.jpg', 'verification': {'status': 'verified'}},
            {'file': '../escape.jpg'}, {'file': 'missing.jpg'}, {'file': 'accepted.jpg'}]}}
        entries = self.api.collect_headshot_artifacts('job', 'synthetic', ['Example'], outcomes)['Example']
        self.assertEqual([entry['filename'] for entry in entries], ['accepted.jpg'])
        self.assertEqual(entries[0]['validation']['status'], 'verified')

    def test_missing_or_abstained_results_never_expose_old_photos(self):
        for outcomes in (None, {}, {'Example': {'status': 'model_unvalidated', 'headshots': [{'file': 'stale.jpg'}]}}):
            self.assertEqual(self.api.collect_headshot_artifacts('job', 'synthetic', ['Example'], outcomes), {'Example': []})

    def test_failed_job_cannot_publish_old_validation_artifacts(self):
        payload = dict(success=False, video_folder='synthetic', headshot_outcomes={
            'Example': {'status': 'accepted', 'headshots': [{'file': 'stale.jpg'}]}})
        result = self.api.finalize_job_result('job', {'actors': ['Example']}, payload)
        self.assertEqual(result['headshots'], {'Example': []})

    def test_json_cli_successfully_reports_optional_no_headshot(self):
        pipeline = script('01_run_headshot_detection.py')
        report = dict(actor='Example', status='no_reliable_headshot', retryable=False, headshots=[])
        (self.video / 'headshot-results.json').write_text(json.dumps({'actors': {'Example': report}}))
        output = io.StringIO()
        with patch('sys.argv', ['pipeline', 'https://example.invalid/video', '--show', 'Fixture', '--actors', 'Example', '--json']), \
                patch.object(pipeline, 'run_actor_training', return_value=True), \
                patch.object(pipeline, 'check_readiness', return_value={}), \
                patch.object(pipeline, 'download_video', return_value=str(self.video)), \
                patch.object(pipeline, 'run_operations_pipeline_with_adaptive_frames', return_value={'Example': 0}), \
                contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as done:
            pipeline.main()
        self.assertEqual(done.exception.code, 0)
        payload = json.loads(output.getvalue().splitlines()[-1])
        self.assertTrue(payload['success'])
        self.assertEqual(payload['outcome'], 'no_reliable_headshots')
        self.assertEqual(payload['headshot_outcomes']['Example']['status'], 'no_reliable_headshot')

    def test_extraction_failure_still_emits_json_error(self):
        pipeline = script('01_run_headshot_detection.py')
        output = io.StringIO()
        with patch('sys.argv', ['pipeline', 'https://example.invalid/video', '--show', 'Fixture', '--actors', 'Example', '--json']), \
                patch.object(pipeline, 'run_actor_training', return_value=True), \
                patch.object(pipeline, 'check_readiness', return_value={}), \
                patch.object(pipeline, 'download_video', return_value=str(self.video)), \
                patch.object(pipeline, 'run_operations_pipeline_with_adaptive_frames', side_effect=RuntimeError('fixture failure')), \
                contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as done:
            pipeline.main()
        self.assertEqual(done.exception.code, 1)
        payload = json.loads(output.getvalue().splitlines()[-1])
        self.assertFalse(payload['success'])
        self.assertEqual(payload['error'], 'fixture failure')


if __name__ == '__main__':
    unittest.main()
