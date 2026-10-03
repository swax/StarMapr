"""Shared headshot crop geometry; no detection or identification dependencies."""

import math

def get_headshot_crop_coordinates(bbox, img_width, img_height):
    """
    Calculate headshot crop coordinates with custom padding and edge detection.

    Args:
        bbox (dict): Bounding box with 'x', 'y', 'w', 'h' keys
        img_width (int): Width of the image
        img_height (int): Height of the image

    Returns:
        dict: Dictionary containing crop coordinates and edge hit flags:
            - x_start, y_start, x_end, y_end: Crop coordinates
            - hit_left_edge, hit_right_edge, hit_top_edge, hit_bottom_edge: Boolean flags
    """
    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']

    # Add padding so headshot looks good
    padding_top = int(h * 0.5)
    padding_bottom = int(h * 1.5)
    padding_left = int(w * 1.5)
    padding_right = int(w * 1.5)

    # Calculate ideal coordinates (before edge constraints)
    x_start_ideal = x - padding_left
    y_start_ideal = y - padding_top
    x_end_ideal = x + w + padding_right
    y_end_ideal = y + h + padding_bottom

    # Apply edge constraints
    x_start = max(0, x_start_ideal)
    y_start = max(0, y_start_ideal)
    x_end = min(img_width, x_end_ideal)
    y_end = min(img_height, y_end_ideal)

    # Determine if we hit any edges
    hit_left_edge = x_start_ideal < 0
    hit_top_edge = y_start_ideal < 0
    hit_right_edge = x_end_ideal > img_width
    hit_bottom_edge = y_end_ideal > img_height
    clipped = hit_left_edge or hit_top_edge or hit_right_edge or hit_bottom_edge

    return {
        'x_start': x_start,
        'y_start': y_start,
        'x_end': x_end,
        'y_end': y_end,
        'clipped': clipped
    }


def select_headshot_crop(bbox, img_width, img_height, min_face_size=50):
    """Prefer normal padding, then reduce only padding around a complete face.

    This selects geometry, never identity. The strict crop helper remains unchanged
    for training and manual extraction. Video operations still apply every model,
    competitor, corroboration and configured cloud gate to either crop mode.
    """
    x, y, w, h = (bbox[key] for key in ('x', 'y', 'w', 'h'))
    values = (x, y, w, h, img_width, img_height, min_face_size)
    if (not all(math.isfinite(value) for value in values)
            or not all(value == int(value) for value in values)
            or min_face_size <= 0 or w < min_face_size or h < min_face_size
            or x < 0 or y < 0 or x + w > img_width or y + h > img_height):
        return None
    x, y, w, h, img_width, img_height, min_face_size = map(int, values)
    bbox = dict(x=x, y=y, w=w, h=h)
    # Match the detector's whole-image false-positive exclusion.
    if abs(w - img_width) <= 3 and abs(h - img_height) <= 3:
        return None
    padded = get_headshot_crop_coordinates(bbox, img_width, img_height)
    bounds = ('x_start', 'y_start', 'x_end', 'y_end')
    if not padded['clipped']:
        return dict(mode='padded', **{key: padded[key] for key in bounds})
    # Clamp padding, never the detected face; no shifting or invented pixels.
    return dict(mode='tight_fallback', fallback_reason='padded_crop_outside_frame',
                x_start=max(0, x - int(w * .25)),
                y_start=max(0, y - int(h * .25)),
                x_end=min(img_width, x + w + int(w * .25)),
                y_end=min(img_height, y + h + int(h * .5)))
