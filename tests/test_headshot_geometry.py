"""Crop fallback preserves visible faces without making an identity decision."""

import unittest

from headshot_geometry import get_headshot_crop_coordinates, select_headshot_crop


class HeadshotGeometryTests(unittest.TestCase):
    def test_normal_crop_keeps_exact_existing_padding(self):
        box = dict(x=400, y=200, w=100, h=80)
        crop = select_headshot_crop(box, 1000, 800)
        self.assertEqual(crop, dict(mode='padded', x_start=250, y_start=160,
                                    x_end=650, y_end=400))
        self.assertFalse(get_headshot_crop_coordinates(box, 1000, 800)['clipped'])

    def test_fallback_preserves_complete_face_at_every_edge(self):
        boxes = [(0, 200, 100, 80), (900, 200, 100, 80), (400, 0, 100, 80),
                 (400, 720, 100, 80), (0, 0, 100, 80), (900, 720, 100, 80)]
        for x, y, w, h in boxes:
            with self.subTest(box=(x, y, w, h)):
                box = dict(x=x, y=y, w=w, h=h)
                self.assertTrue(get_headshot_crop_coordinates(box, 1000, 800)['clipped'])
                crop = select_headshot_crop(box, 1000, 800)
                self.assertEqual(crop['mode'], 'tight_fallback')
                self.assertEqual(crop['fallback_reason'], 'padded_crop_outside_frame')
                self.assertLessEqual(0, crop['x_start'])
                self.assertLessEqual(0, crop['y_start'])
                self.assertLessEqual(crop['x_start'], x)
                self.assertLessEqual(crop['y_start'], y)
                self.assertGreaterEqual(crop['x_end'], x + w)
                self.assertGreaterEqual(crop['y_end'], y + h)
                self.assertLessEqual(crop['x_end'], 1000)
                self.assertLessEqual(crop['y_end'], 800)
                self.assertLess(crop['x_end'] - crop['x_start'], 4 * w)

    def test_partial_small_and_false_positive_faces_remain_ineligible(self):
        boxes = [(-1, 200, 100, 80), (901, 200, 100, 80), (400, -1, 100, 80),
                 (400, 721, 100, 80), (400, 200, 49, 80), (400, 200, 100, 49),
                 (0, 0, 1000, 800), (400.5, 200, 100, 80), (float('nan'), 200, 100, 80)]
        for x, y, w, h in boxes:
            with self.subTest(box=(x, y, w, h)):
                self.assertIsNone(select_headshot_crop(dict(x=x, y=y, w=w, h=h), 1000, 800))


if __name__ == '__main__':
    unittest.main()
