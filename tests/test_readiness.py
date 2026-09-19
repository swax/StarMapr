"""Synthetic-only readiness, migration, progress and durable budget checks."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from test_api_validation import script
from celebrity_verifier import CelebrityVerifier
from progress import run_logged


class ReadinessTests(unittest.TestCase):
    def test_metadata_only_download_cache_is_not_ready(self):
        module = script('30_download_video.py')
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'info.json').write_text('{}')
            Path(tmp, 'thumbnail.jpg').write_bytes(b'synthetic')
            self.assertFalse(module.has_playable_video(tmp))

    def test_api_progress_stream_does_not_replace_final_json(self):
        module = script('90_run_api_server.py')
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp)/'job.log'
            command = [sys.executable, '-c', 'print(\'STARMAPR_PROGRESS {"phase":"fixture","state":"started"}\'); print(\'{"success":true}\')']
            with log.open('w') as handle, patch.object(module, 'ROOT_DIR', Path(tmp)), \
                    patch.object(module, 'get_job_log_path', return_value=log), \
                    patch.object(module, 'update_job') as update:
                code, last = module.run_command_to_log(command, handle, 'fixture')
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(last)['success'])
            update.assert_called_once_with('fixture', progress={'phase': 'fixture', 'state': 'started'})

    def test_legacy_model_automatically_attempts_migration_and_preserves_model(self):
        module = script('02_run_actor_training.py')
        with patch('sys.argv', ['train', 'Example', 'Fixture']), \
             patch.object(module, 'reusable_model', return_value=False), \
             patch.object(module, 'check_existing_model', return_value=True), \
             patch.object(module, 'delete_existing_folders') as archive, \
             patch.object(module, 'run_logged', side_effect=subprocess.CalledProcessError(2, ['fixture'])) as run, \
             patch.object(module, 'copy_model_to_models_dir') as promote, \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as done:
            module.main()
        self.assertEqual(done.exception.code, 0)
        archive.assert_called_once_with('Example')
        run.assert_called_once()
        promote.assert_not_called()

    def test_valid_model_does_not_retrain(self):
        module = script('02_run_actor_training.py')
        with patch('sys.argv', ['train', 'Example', 'Fixture']), \
             patch.object(module, 'reusable_model', return_value=True), \
             patch.object(module, 'delete_existing_folders') as archive, \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            module.main()
        archive.assert_not_called()

    def test_preflight_failure_does_not_train_or_download(self):
        module = script('01_run_headshot_detection.py')
        with patch('sys.argv', ['run', 'https://example.invalid', '--actors', 'Example', '--show', 'Fixture', '--json']), \
             patch.object(module, 'check_readiness', side_effect=RuntimeError('Preflight: expired')), \
             patch.object(module, 'run_actor_training') as train, \
             patch.object(module, 'download_video') as download, \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as done:
            module.main()
        self.assertEqual(done.exception.code, 1)
        train.assert_not_called()
        download.assert_not_called()

    def test_actor_share_is_durable_and_does_not_starve_another_actor(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = Mock()
            client.recognize_celebrities.return_value = {'CelebrityFaces': []}
            opts = dict(max_requests=2, actor_limits={'one': 1, 'two': 1}, client=client)
            first = CelebrityVerifier(Path(tmp)/'cache.sqlite', **opts)
            first.verify(b'synthetic-one', expected_name='One')
            second = CelebrityVerifier(Path(tmp)/'cache.sqlite', **opts)
            self.assertEqual(second.verify(b'synthetic-two', expected_name='One')['status'], 'actor_budget_exhausted')
            self.assertEqual(second.verify(b'synthetic-three', expected_name='Two')['status'], 'unknown_or_multiple_faces')
            self.assertEqual(client.recognize_celebrities.call_count, 2)

    def test_progress_retains_full_log_and_bounds_error_tail(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'STARMAPR_PROGRESS_FILE': str(Path(tmp)/'progress.json')}):
            command = [sys.executable, '-c', 'import sys; print("x" * 25000); print("distinct failure", file=sys.stderr); sys.exit(7)']
            with self.assertRaises(subprocess.CalledProcessError) as failed:
                run_logged(command, check=True)
            self.assertLessEqual(len(failed.exception.stdout), 16000)
            self.assertIn('distinct failure', failed.exception.stderr)
            status = json.loads((Path(tmp)/'progress.json').read_text())
            self.assertEqual(status['exit_code'], 7)
            self.assertGreater(Path(status['log']).stat().st_size, 25000)

    def test_archive_retains_budget_and_original_data(self):
        module = script('02_run_actor_training.py')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode in ('training', 'testing'):
                folder = root/mode/'example'
                folder.mkdir(parents=True)
                (folder/'original.txt').write_text('preserve')
            (root/'training/example/celebrity-cache.sqlite').write_bytes(b'budget-fixture')
            with patch.object(module, 'get_actor_folder_path', side_effect=lambda _, mode: root/mode/'example'):
                module.delete_existing_folders('Example')
            self.assertEqual((root/'training/example/celebrity-cache.sqlite').read_bytes(), b'budget-fixture')
            self.assertEqual(len(list(root.glob('*/.history/*/original.txt'))), 2)
