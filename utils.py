"""
StarMapr Utilities Module

Common utility functions used across multiple scripts in the StarMapr project.
This module consolidates repetitive code patterns to improve maintainability
and consistency.
"""

import os
import pickle
import argparse
import unicodedata
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_actor_folder_path(actor_name, mode='training'):
    """
    Convert actor name to folder path with consistent naming convention.
    
    Args:
        actor_name (str): The actor name
        mode (str): Either 'training' or 'testing'
        
    Returns:
        str: Formatted folder path (e.g., '02_training/bill_murray/')
    """
    actor_folder = get_actor_folder_name(actor_name)
    mode_prefix = '02_' if mode == 'training' else '03_'
    return f'{mode_prefix}{mode}/{actor_folder}/'


def get_actor_folder_name(actor_name):
    """
    Get just the folder name part from actor name.
    
    Args:
        actor_name (str): The actor name
        
    Returns:
        str: Formatted folder name (e.g., 'bill_murray')
    """
    normalized = unicodedata.normalize('NFKD', actor_name)
    ascii_name = normalized.encode('ascii', 'ignore').decode('ascii')
    return ascii_name.lower().replace(' ', '_')


def get_average_embedding_filename(actor_name):
    """
    Generate the standard filename for average embedding files.
    
    Args:
        actor_name (str): The actor name
        
    Returns:
        str: Formatted filename (e.g., 'bill_murray_average_embedding.pkl')
    """
    actor_folder = get_actor_folder_name(actor_name)
    return f"{actor_folder}_average_embedding.pkl"


def get_average_embedding_path(actor_name, location='training'):
    """
    Generate the full path to the average embedding file.
    
    Args:
        actor_name (str): The actor name
        location (str): Either 'training' for training folder or 'models' for models folder
        
    Returns:
        Path: Full path to the average embedding file
    """
    filename = get_average_embedding_filename(actor_name)
    
    if location == 'training':
        base_path = Path(get_actor_folder_path(actor_name, 'training'))
    elif location == 'models':
        base_path = Path("04_models")
    else:
        raise ValueError(f"Invalid location: {location}. Must be 'training' or 'models'")
    
    return base_path / filename


def add_training_testing_args(parser):
    """
    Add standard --training/--testing mutually exclusive arguments to parser.
    
    Args:
        parser (argparse.ArgumentParser): The argument parser
        
    Returns:
        argparse.ArgumentParser: The modified parser
    """
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--training', metavar='ACTOR_NAME',
                           help='Process 02_training/ACTOR_NAME/ folder')
    mode_group.add_argument('--testing', metavar='ACTOR_NAME', 
                           help='Process 03_testing/ACTOR_NAME/ folder')
    return parser


def get_mode_and_path_from_args(args):
    """
    Extract mode and actor folder path from parsed arguments.
    
    Args:
        args: Parsed arguments from argparse
        
    Returns:
        tuple: (mode, actor_name, folder_path)
    """
    if args.training:
        actor_name = args.training
        mode = 'training'
    else:
        actor_name = args.testing
        mode = 'testing'
    
    folder_path = get_actor_folder_path(actor_name, mode)
    return mode, actor_name, folder_path


def get_image_files(folder_path, exclude_subdirs=True):
    """
    Get all image files from folder with standard extensions.
    
    Args:
        folder_path (str or Path): Path to the folder
        exclude_subdirs (bool): Whether to exclude subdirectories
        
    Returns:
        list: List of Path objects for image files
    """
    image_extensions = get_supported_image_extensions()
    folder_path = Path(folder_path)
    
    if exclude_subdirs:
        return [f for f in folder_path.iterdir() 
                if f.is_file() and f.suffix.lower() in image_extensions]
    else:
        return [f for f in folder_path.rglob('*') 
                if f.is_file() and f.suffix.lower() in image_extensions]




def get_env_int(key, default):
    """
    Get integer value from environment with default.
    
    Args:
        key (str): Environment variable key
        default (int): Default value if key not found
        
    Returns:
        int: Environment value or default
    """
    return int(os.getenv(key, default))


def get_env_float(key, default):
    """
    Get float value from environment with default.
    
    Args:
        key (str): Environment variable key
        default (float): Default value if key not found
        
    Returns:
        float: Environment value or default
    """
    return float(os.getenv(key, default))


def get_default_thresholds():
    """
    Get all default threshold values from environment.
    
    Returns:
        dict: Dictionary of all threshold and count values
    """
    return {
        'training_duplicate': get_env_int('TRAINING_DUPLICATE_THRESHOLD', 5),
        'training_outlier': get_env_float('TRAINING_OUTLIER_THRESHOLD', 0.1),
        'testing_detection': get_env_float('TESTING_DETECTION_THRESHOLD', 0.6),
        'headshot_match': get_env_float('OPERATIONS_HEADSHOT_MATCH_THRESHOLD', 0.6),
        'extract_frame_count': get_env_int('OPERATIONS_EXTRACT_FRAME_COUNT', 50)
    }


# ANSI color codes
class Colors:
    RED = '\033[91m'
    PURPLE = '\033[95m'
    RESET = '\033[0m'


def print_error(message):
    """
    Print error message in red color.
    
    Args:
        message (str): Error message to print
    """
    print(f"{Colors.RED}{message}{Colors.RESET}")


def print_summary(message):
    """
    Print summary message in purple color.
    
    Args:
        message (str): Summary message to prints
    """
    print(f"{Colors.PURPLE}{message}{Colors.RESET}")


def print_dry_run_header(action_description):
    """
    Print consistent dry-run header.
    
    Args:
        action_description (str): Description of what would be done
    """
    print(f"DRY RUN MODE - {action_description}")


def print_dry_run_summary(would_count, action_description):
    """
    Print consistent dry-run summary.
    
    Args:
        would_count (int): Number of items that would be affected
        action_description (str): Description of the action
    """
    print(f"DRY RUN: Would {action_description} {would_count} items")


def log(message = ""):
    """
    Log function to replace standard print statements.
    Only prints if STARMAPR_LOG_VERBOSE environment variable is set.
    
    Args:
        message (str): Message to log
    """
    if os.getenv('STARMAPR_LOG_VERBOSE'):
        print(message)


def save_pickle(data, file_path):
    """
    Save data to pickle file with error handling.
    
    Args:
        data: Data to save
        file_path (str or Path): Path to save file
        
    Returns:
        bool: True if successful, False if error
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print_error(f"Error saving to {file_path}: {e}")
        return False


def load_pickle(file_path):
    """
    Load data from pickle file with error handling.
    
    Args:
        file_path (str or Path): Path to pickle file
        
    Returns:
        object or None: Loaded data, None if error
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print_error(f"Error loading from {file_path}: {e}")
        return None


def ensure_folder_exists(folder_path):
    """
    Ensure a folder exists, create if it doesn't.
    
    Args:
        folder_path (str or Path): Path to the folder
        
    Returns:
        Path: Path object for the folder
    """
    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def calculate_face_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two face embeddings.
    
    Args:
        embedding1: First face embedding
        embedding2: Second face embedding
        
    Returns:
        float: Cosine similarity score (0.0 to 1.0)
    """
    emb1 = np.array(embedding1).reshape(1, -1)
    emb2 = np.array(embedding2).reshape(1, -1)
    return cosine_similarity(emb1, emb2)[0][0]


def get_supported_image_extensions():
    """
    Get the set of supported image file extensions.
    
    Returns:
        set: Set of supported extensions (with dots)
    """
    return {'.gif', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def get_corresponding_pkl_file(image_path):
    """
    Get the corresponding pkl file path for an image file.
    
    Args:
        image_path (str or Path): Path to the image file
        
    Returns:
        Path or None: Path to corresponding pkl file if it exists, None otherwise
    """
    image_path = Path(image_path)
    pkl_path = image_path.with_suffix('.pkl')
    return pkl_path if pkl_path.exists() else None


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


def move_file_with_pkl(source_path, destination_folder, dry_run=False):
    """
    Move a file and its corresponding pkl file (if it exists) to destination folder.
    
    Args:
        source_path (str or Path): Path to the source file
        destination_folder (str or Path): Destination folder
        dry_run (bool): If True, only report what would be moved
        
    Returns:
        tuple: (moved_files_count, total_files_attempted)
    """
    import shutil
    
    source_path = Path(source_path)
    destination_folder = Path(destination_folder)
    moved_count = 0
    attempted_count = 0
    
    # Move the main file
    try:
        if not dry_run:
            destination_folder.mkdir(parents=True, exist_ok=True)
            destination = destination_folder / source_path.name
            shutil.move(str(source_path), str(destination))
            log(f"  ✓ Moved: {source_path.name}")
        else:
            log(f"  → Would move: {source_path.name}")
        moved_count += 1
        attempted_count += 1
    except Exception as e:
        print_error(f"Failed to move {source_path.name}: {e}")
        attempted_count += 1
    
    # Move corresponding pkl file if it exists
    pkl_path = get_corresponding_pkl_file(source_path)
    if pkl_path:
        try:
            if not dry_run:
                pkl_destination = destination_folder / pkl_path.name
                shutil.move(str(pkl_path), str(pkl_destination))
                log(f"  ✓ Moved: {pkl_path.name}")
            else:
                log(f"  → Would move: {pkl_path.name}")
            moved_count += 1
            attempted_count += 1
        except Exception as e:
            print_error(f"Failed to move {pkl_path.name}: {e}")
            attempted_count += 1
    
    return moved_count, attempted_count
