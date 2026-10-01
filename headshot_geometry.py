"""Shared headshot crop geometry; no detection or identification dependencies."""

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
