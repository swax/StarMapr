"""Offline regression tests for paid-search avoidance and accepted image provenance."""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from image_sources import MANIFEST, accepted_sources, domain, save_search_sources
from training_history import COOLDOWN_SECONDS, TrainingHistory
from validation import QualityPolicy, file_hash, metadata_path, write_json

ROOT = Path(__file__).resolve().parents[1]


def script(filename):
    spec = importlib.util.spec_from_file_location(filename.removesuffix('.py'), ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TemporaryHistory(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.history = TrainingHistory(self.folder / 'history.sqlite')


class HistoryTests(TemporaryHistory):
    def test_seven_days_survive_restart_and_skips_do_not_extend_deadline(self):
        until = self.history.record_failure('Example', 'testing', now=100)
        restarted = TrainingHistory(self.history.path)
        self.assertEqual(restarted.skip_if_cooling_down('Example', now=101)['retry_after'], until)
        self.assertIsNone(restarted.skip_if_cooling_down('Other', now=101))
        self.assertEqual(restarted.skip_if_cooling_down('Example', now=99 + COOLDOWN_SECONDS)['cooldown_skips'], 2)
        self.assertIsNone(restarted.skip_if_cooling_down('Example', now=100 + COOLDOWN_SECONDS))
        report = restarted.summary(now=100 + COOLDOWN_SECONDS)
        self.assertEqual((report['failed_runs'], report['cooldown_skips'], report['active_cooldowns']), (1, 2, 0))

    def test_concurrent_skips_are_all_counted(self):
        self.history.record_failure('Example', 'training', now=100)
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda _: self.history.skip_if_cooling_down('Example', now=101), range(18)))
        self.assertEqual(self.history.summary(now=101)['cooldown_skips'], 18)

    def test_success_clears_cooldown_and_domain_counts_deduplicate_retrains(self):
        self.history.record_failure('Example', 'training', now=100)
        self.history.skip_if_cooling_down('Example', now=101)
        report = dict(model_sha256='model', training_images={'good.jpg': 'good', 'manual.jpg': 'manual'},
                      training_sources=[dict(filename='good.jpg', sha256='good', page_domain='example.org',
                                             image_domain='cdn.example.net'),
                                        dict(filename='rejected.jpg', sha256='bad', page_domain='wrong.org')])
        self.history.record_success('Example', 'Show', report, now=102)
        self.history.record_success('Example', 'Show', report, now=103)
        self.assertIsNone(self.history.skip_if_cooling_down('Example', now=104))
        stats = self.history.summary(now=104)
        self.assertEqual(stats['page_domain'], [dict(domain='example.org', images=1, actors=1, successful_runs=2)])
        self.assertEqual(stats['image_domain'][0]['domain'], 'cdn.example.net')
        self.assertEqual(stats['accepted_image_uses_without_page_domain'], 2)
        self.assertEqual((stats['failed_runs'], stats['cooldown_skips'], stats['successful_runs']), (1, 1, 2))
        self.history.record_failure('Example', 'testing', now=105)
        self.assertIsNotNone(self.history.skip_if_cooling_down('Example', now=106))

    def test_mismatched_source_hash_is_not_counted(self):
        self.history.record_success('Example', 'Show', dict(model_sha256='model', training_images={'a.jpg': 'new'},
            training_sources=[dict(filename='a.jpg', sha256='old', page_domain='wrong.org')]))
        self.assertEqual(self.history.summary()['page_domain'], [])

    def test_report_cli_exports_all_source_page_domains(self):
        stats = script('93_training_stats.py')
        output = self.folder / 'domains.csv'
        self.history.record_success('Example', 'Show', dict(model_sha256='model', training_images={'a.jpg': 'hash'},
            training_sources=[dict(filename='a.jpg', sha256='hash', page_domain='example.org')]))
        captured = io.StringIO()
        with patch.object(stats, 'TrainingHistory', return_value=self.history), \
                patch('sys.argv', ['stats', '--json', '--domains-csv', str(output)]), \
                contextlib.redirect_stdout(captured):
            stats.main()
        self.assertEqual(json.loads(captured.getvalue())['successful_runs'], 1)
        self.assertIn('example.org,1,1,1', output.read_text())


class OrchestrationTests(TemporaryHistory):
    def setUp(self):
        super().setUp()
        self.training = script('02_run_actor_training.py')
        for name, replacement in (
            ('TrainingHistory', Mock(return_value=self.history)), ('reusable_model', Mock(return_value=False)),
            ('check_existing_model', Mock(return_value=False)), ('delete_existing_folders', Mock()),
            ('emit_progress', Mock()), ('get_venv_python', Mock(return_value='python'))):
            patcher = patch.object(self.training, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)

    def run_main(self, *extra):
        with patch('sys.argv', ['02_run_actor_training.py', 'Example', 'Show', *extra]), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as exited:
            self.training.main()
        return exited.exception.code

    def test_cooldown_prevents_search_and_archiving_even_with_retrain(self):
        self.history.record_failure('Example', 'training')
        with patch.object(self.training, 'run_logged') as run:
            self.assertEqual(self.run_main('--retrain'), 0)
        run.assert_not_called()
        self.training.delete_existing_folders.assert_not_called()
        self.assertEqual(self.history.summary()['cooldown_skips'], 1)

    def test_usable_model_is_reused_without_counting_a_cooldown(self):
        self.history.record_failure('Example', 'training')
        self.training.reusable_model.return_value = True
        with patch.object(self.training, 'run_logged') as run:
            self.assertEqual(self.run_main(), 0)
        run.assert_not_called()
        self.assertEqual(self.history.summary()['cooldown_skips'], 0)

    def test_quality_abstention_and_training_errors_start_cooldown(self):
        for code in (1, 2):
            with self.subTest(code=code), patch.object(self.training, 'run_logged',
                    side_effect=subprocess.CalledProcessError(code, ['training'])):
                self.assertEqual(self.run_main('--ignore-cooldown'), 0 if code == 2 else 1)
            self.assertEqual(self.history.summary()['actors'][0]['failure_phase'], 'training')
        self.assertEqual(self.history.summary()['failed_runs'], 2)

    def test_testing_failure_starts_cooldown(self):
        with patch.object(self.training, 'run_logged', side_effect=[SimpleNamespace(stdout=''),
                subprocess.CalledProcessError(1, ['testing'])]), \
                patch.object(self.training, 'copy_model_to_models_dir') as promote:
            self.assertEqual(self.run_main(), 1)
        promote.assert_not_called()
        self.assertEqual(self.history.summary()['actors'][0]['failure_phase'], 'testing')

    def test_promotion_failure_does_not_record_domains(self):
        with patch.object(self.training, 'run_logged', return_value=SimpleNamespace(stdout='')), \
                patch.object(self.training, 'copy_model_to_models_dir', return_value=False):
            self.assertEqual(self.run_main(), 1)
        report = self.history.summary()
        self.assertEqual(report['actors'][0]['failure_phase'], 'promotion')
        self.assertEqual(report['successful_runs'], 0)

    def test_success_records_only_after_promotion_and_explicit_override_works(self):
        self.history.record_failure('Example', 'testing')
        model = self.folder / 'model.pkl'
        write_json(metadata_path(model), dict(model_sha256='model', training_images={}))
        with patch.object(self.training, 'run_logged', return_value=SimpleNamespace(stdout='')), \
                patch.object(self.training, 'copy_model_to_models_dir', return_value=True), \
                patch.object(self.training, 'get_average_embedding_path', return_value=model):
            self.assertEqual(self.run_main('--ignore-cooldown'), 0)
        report = self.history.summary()
        self.assertEqual((report['successful_runs'], report['active_cooldowns'], report['cooldown_skips']), (1, 0, 0))

    def test_process_launch_error_also_cools_down(self):
        with patch.object(self.training, 'run_logged', side_effect=OSError('could not launch worker')):
            self.assertEqual(self.run_main(), 1)
        self.assertEqual(self.history.summary()['failed_runs'], 1)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.downloader = script('10_download_actor_images.py')

    def test_download_records_urls_and_cache_reuse_never_searches(self):
        previous = Path.cwd()
        os.chdir(self.folder)
        self.addCleanup(os.chdir, previous)
        gis = Mock()
        def search(*, search_params, path_to_dir):
            image = Path(path_to_dir) / 'photo.jpg'
            image.write_bytes(b'\xff\xd8\xffoffline fixture')
            gis.results.return_value = [SimpleNamespace(path=str(image), url='https://cdn.example.org/photo.jpg',
                                                        referrer_url='https://example.org/actor')]
        gis.search.side_effect = search
        with patch.object(self.downloader, 'GoogleImagesSearch', return_value=gis) as factory:
            for _ in range(2):
                self.assertTrue(self.downloader.download_actor_images('Example', show='Show',
                                                                      api_key='fixture', search_engine_id='fixture'))
        factory.assert_called_once()
        gis.search.assert_called_once()
        sources = accepted_sources(Path('02_training/example').glob('*.jpg'))
        self.assertEqual(len(sources), 2)
        self.assertTrue(all(row['page_domain'] == 'example.org' for row in sources))

    def test_sources_survive_guid_renaming_and_cached_reuse(self):
        cache = self.folder / 'cache'
        cache.mkdir()
        photo = cache / 'photo.jpg'
        photo.write_bytes(b'\xff\xd8\xffsynthetic')
        rejected = cache / 'rejected.jpg'
        rejected.write_bytes(b'\xff\xd8\xffrejected')
        result = SimpleNamespace(path=str(photo), url='https://cdn.example.net/photo.jpg',
                                 referrer_url='https://www.example.org/actor')
        save_search_sources(cache, [result], 'Example Show')
        for attempt in ('first', 'cached'):
            destination = self.folder / attempt
            self.assertEqual(self.downloader.copy_images_from_cache_to_destination(cache, destination), 2)
            images = [image for image in destination.glob('*.jpg') if file_hash(image) == file_hash(photo)]
            sources = accepted_sources(images)
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0]['page_domain'], 'example.org')
            self.assertEqual(sources[0]['image_domain'], 'cdn.example.net')
            self.assertEqual(sources[0]['query'], 'Example Show')
            self.assertNotEqual(sources[0]['filename'], photo.name)
            images[0].write_bytes(b'changed bytes')
            self.assertIsNone(accepted_sources(images)[0]['page_domain'])

    def test_colliding_cache_filenames_have_unknown_source(self):
        photo = self.folder / 'photo.jpg'
        photo.write_bytes(b'synthetic')
        results = [SimpleNamespace(path=str(photo), url=f'https://{name}.org/photo.jpg',
                                   referrer_url=f'https://{name}.org/') for name in ('one', 'two')]
        save_search_sources(self.folder, results, 'Example')
        self.assertIsNone(accepted_sources([photo])[0]['page_domain'])

    def test_only_images_contributing_embeddings_enter_manifest(self):
        embedding = script('15_compute_average_embeddings.py')
        for name in ('one', 'two', 'three', 'no_face'):
            (self.folder / f'{name}.jpg').write_bytes(name.encode())
        def faces(path):
            return [] if path.stem == 'no_face' else [{'embedding': [1., 0., 0.]}]
        with patch.object(embedding, 'get_face_embeddings', side_effect=faces):
            _, count = embedding.compute_average_embeddings(self.folder, QualityPolicy(3))
        report = json.loads((self.folder / 'training-quality.json').read_text())
        self.assertEqual(count, 3)
        self.assertEqual({row['filename'] for row in report['training_sources']}, {'one.jpg', 'two.jpg', 'three.jpg'})

    def test_domain_normalization_keeps_meaningful_subdomains(self):
        self.assertEqual(domain('https://WWW.Example.CO.UK./image'), 'example.co.uk')
        self.assertEqual(domain('https://images.example.org/x'), 'images.example.org')
        for url in ('file:///photo', 'not a url', 'https://[invalid'):
            self.assertIsNone(domain(url))


if __name__ == '__main__':
    unittest.main()
