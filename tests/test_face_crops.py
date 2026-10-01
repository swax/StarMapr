import importlib.util
import unittest
import argparse
import tempfile
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("face_crops", Path(__file__).resolve().parents[1] / "35_extract_face_crops.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FaceCropGeometryTests(unittest.TestCase):
    def test_crop_stays_square_and_inside_frame_at_every_edge(self):
        for box in [(0, 0, 100, 90), (900, 0, 100, 100), (0, 480, 100, 100), (900, 480, 100, 100)]:
            left, top, right, bottom = module.crop_bounds(box, 1000, 580, 1.7)
            self.assertEqual(right - left, bottom - top)
            self.assertGreaterEqual(left, 0)
            self.assertGreaterEqual(top, 0)
            self.assertLessEqual(right, 1000)
            self.assertLessEqual(bottom, 580)

    def test_sampling_never_seeks_to_or_beyond_video_end(self):
        self.assertEqual(module.sample_times(10, 0, None, 5), [0, 5])
        self.assertEqual(module.sample_times(10, 3, 50, 4), [3, 7])

    def test_invalid_or_unbounded_sampling_is_rejected(self):
        for interval in [0, -1, float("nan"), float("inf"), 0.00001]:
            with self.assertRaises(ValueError):
                module.sample_times(100, 0, None, interval)
        for timestamp in [-1, 10, float("nan")]:
            with self.assertRaises(ValueError):
                module.sample_times(10, 0, None, 1, [timestamp])

    def test_explicit_timestamps_are_sorted_and_deduplicated(self):
        self.assertEqual(module.sample_times(10, 0, None, 1, [8, 2, 8]), [2, 8])


class FaceCropRuntimeTests(unittest.TestCase):
    def test_blank_video_is_decoded_without_inventing_faces(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "blank.avi"
            output = Path(tmp) / "output"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
            self.assertTrue(writer.isOpened())
            for _ in range(10):
                writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
            writer.release()
            args = argparse.Namespace(video=video, output=output, start=0, end=None, interval=0.5, timestamps=None, min_face=72, padding=1.7)
            self.assertEqual(module.extract(args), 0)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(manifest["frames"]), 2)
            self.assertEqual(manifest["crops"], [])
            self.assertEqual(manifest["skipped_timestamps"], [])
            self.assertTrue(manifest["review_required"])
            with self.assertRaisesRegex(ValueError, "must be empty"):
                module.extract(args)


if __name__ == "__main__":
    unittest.main()
