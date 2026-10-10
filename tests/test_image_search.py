"""Exercise the installed Google SDK's batch behavior without any network calls."""

import contextlib
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from image_search import SourceRecordingGoogleImagesSearch
from image_sources import accepted_sources, read_sources, save_search_sources


class ImageSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        encoded = io.BytesIO()
        Image.new('RGB', (8, 8), 'red').save(encoded, format='PNG')
        self.pixels = encoded.getvalue()
        self.search = SourceRecordingGoogleImagesSearch('fixture', 'fixture', validate_images=False)
        # Bypass all remote services; still execute the SDK's real download, batching,
        # threading, conversion and result-truncation implementation.
        self.search.get_raw_data = Mock(side_effect=lambda url: iter([self.pixels]))
        self.backend = Mock()
        self.search._google_custom_search = self.backend

    def results(self, start, count):
        return [(f'https://images.example.org/photo.jpg?id={i}', f'https://pages.example.org/actor/{i}')
                for i in range(start, start + count)]

    def test_overfilled_batch_keeps_all_urls_and_same_basename_images(self):
        self.backend.search.side_effect = [self.results(0, 8), self.results(8, 8), self.results(16, 8)]
        self.search.search(search_params={'q': 'Example', 'num': 20}, path_to_dir=str(self.folder))
        self.assertEqual(len(self.search.results()), 20)  # the original loss mechanism
        downloaded = self.search.downloaded_results()
        self.assertEqual(len(downloaded), 24)
        self.assertEqual(len({image.path for image in downloaded}), 24)
        self.assertEqual(len(list(self.folder.glob('*.jpg'))), 24)
        save_search_sources(self.folder, downloaded, 'Example')
        records = accepted_sources(self.folder.glob('*.jpg'))
        self.assertEqual(len(records), 24)
        self.assertTrue(all(row['page_domain'] == 'pages.example.org' for row in records))
        self.assertEqual({row['image_url'] for row in records}, {url for url, _ in self.results(0, 24)})
        self.assertEqual(self.backend.search.call_count, 3)
        self.assertFalse(any(path.is_dir() for path in self.folder.iterdir()))

    def test_partial_search_failure_still_saves_completed_source_urls(self):
        spec = importlib.util.spec_from_file_location('downloader', Path(__file__).resolve().parents[1] / '10_download_actor_images.py')
        downloader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(downloader)
        previous = Path.cwd()
        os.chdir(self.folder)
        self.addCleanup(os.chdir, previous)
        self.backend.search.side_effect = [self.results(0, 8), RuntimeError('fixture search failure')]
        with patch.object(downloader, 'GoogleImagesSearch', return_value=self.search), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(downloader.download_actor_images('Example', show='Show',
                                                              api_key='fixture', search_engine_id='fixture'))
        cache = Path('01_images/example/training/example_show')
        self.assertEqual(len(read_sources(cache)), 8)
        self.assertTrue(all(row['page_domain'] for row in accepted_sources(cache.glob('*.jpg'))))

    def test_headshot_exclusion_checks_original_url_even_when_filename_is_shortened(self):
        spec = importlib.util.spec_from_file_location('downloader', Path(__file__).resolve().parents[1] / '10_download_actor_images.py')
        downloader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(downloader)
        url = 'https://example.org/actor_match_0.52_position_1234_' + 'x' * 170 + '.jpg'
        self.backend.search.return_value = [(url, 'https://example.org/sketch')]
        self.search.search(search_params={'q': 'Example', 'num': 1}, path_to_dir=str(self.folder))
        save_search_sources(self.folder, self.search.downloaded_results(), 'Example')
        self.assertEqual(downloader.copy_images_from_cache_to_destination(self.folder, self.folder / 'training'), 0)


if __name__ == '__main__':
    unittest.main()
