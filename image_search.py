"""Google image downloads with complete provenance and collision-free cache names."""

import hashlib
import os
import tempfile
import threading
from pathlib import Path

from google_images_search import GoogleImagesSearch


class SourceRecordingGoogleImagesSearch(GoogleImagesSearch):
    """Keep every completed download before the SDK truncates its results to `num`.

    The SDK downloads concurrent batches and can save more images than requested.
    Its ordinary results() omits those extra images even though they remain in the
    cache. Capture each image object at completion so their URLs are not lost.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._source_results = []
        self._source_lock = threading.Lock()

    def _download_and_resize(self, path_to_dir, image, width, height):
        super()._download_and_resize(path_to_dir, image, width, height)
        with self._source_lock:
            self._source_results.append(image)

    def downloaded_results(self):
        with self._source_lock:
            return list(self._source_results)

    def download(self, url, path_to_dir):
        folder = Path(path_to_dir)
        folder.mkdir(parents=True, exist_ok=True)
        # Different source URLs often share names such as image.jpg or hq720.jpg.
        # Isolate the SDK's write, then publish a stable URL-specific filename.
        with tempfile.TemporaryDirectory(prefix='.download-', dir=folder) as temporary:
            downloaded = Path(super().download(url, temporary))
            prefix = hashlib.sha256(url.encode('utf-8')).hexdigest()[:24]
            destination = folder / f'{prefix}-{downloaded.name[-150:]}'
            os.replace(downloaded, destination)
        return str(destination)
