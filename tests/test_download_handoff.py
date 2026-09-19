import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("download", Path(__file__).resolve().parents[1] / "30_download_video.py")
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)

class DownloadHandoffTests(unittest.TestCase):
    def test_absolute_paths_and_literal_names(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                folder = Path("05_videos/youtube_synthetic")
                folder.mkdir(parents=True)
                (folder / "A$AP Rocky.webm").write_bytes(b"synthetic fixture")
                result = download.build_download_result("https://youtu.be/synthetic", "youtube", "synthetic")
                self.assertEqual(result["video_folder"], str(folder.resolve()))
                self.assertEqual(result["files"], [{"path": str((folder / "A$AP Rocky.webm").resolve()), "bytes": 17}])
                self.assertTrue(result["host"])
                os.chdir(previous)
                self.assertTrue(Path(result["files"][0]["path"]).exists())
            finally:
                os.chdir(previous)
