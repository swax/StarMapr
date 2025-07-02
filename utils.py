"""
StarMapr Utilities Module

Common utility functions used across multiple scripts in the StarMapr project.
This module consolidates repetitive code patterns to improve maintainability
and consistency.
"""

import os
import pickle
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_celebrity_folder_path(celebrity_name, mode='training'):
    """
    Convert celebrity name to folder path with consistent naming convention.
    
    Args:
        celebrity_name (str): The celebrity name
        mode (str): Either 'training' or 'testing'
        
    Returns:
        str: Formatted folder path (e.g., 'training/bill_murray/')
    """
    celebrity_folder = celebrity_name.lower().replace(' ', '_')
    return f'{mode}/{celebrity_folder}/'


def get_celebrity_folder_name(celebrity_name):
    """
    Get just the folder name part from celebrity name.
    
    Args:
        celebrity_name (str): The celebrity name
        
    Returns:
        str: Formatted folder name (e.g., 'bill_murray')
    """
    return celebrity_name.lower().replace(' ', '_')


def add_training_testing_args(parser):
    """
    Add standard --training/--testing mutually exclusive arguments to parser.
    
    Args:
        parser (argparse.ArgumentParser): The argument parser
        
    Returns:
        argparse.ArgumentParser: The modified parser
    """
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--training', metavar='CELEBRITY_NAME',
                           help='Process training/CELEBRITY_NAME/ folder')
    mode_group.add_argument('--testing', metavar='CELEBRITY_NAME', 
                           help='Process testing/CELEBRITY_NAME/ folder')
    return parser


def get_mode_and_path_from_args(args):
    """
    Extract mode and celebrity folder path from parsed arguments.
    
    Args:
        args: Parsed arguments from argparse
        
    Returns:
        tuple: (mode, celebrity_name, folder_path)
    """
    if args.training:
        celebrity_name = args.training
        mode = 'training'
    else:
        celebrity_name = args.testing
        mode = 'testing'
    
    folder_path = get_celebrity_folder_path(celebrity_name, mode)
    return mode, celebrity_name, folder_path


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
        'training_image_count': get_env_int('TRAINING_IMAGE_COUNT', 20),
        'testing_image_count': get_env_int('TESTING_IMAGE_COUNT', 30),
        'extract_frame_count': get_env_int('OPERATIONS_EXTRACT_FRAME_COUNT', 50)
    }


# ANSI color codes
class Colors:
    RED = '\033[91m'
    BLUE = '\033[94m'
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
    Print summary message in blue color.
    
    Args:
        message (str): Summary message to print
    """
    print(f"{Colors.BLUE}{message}{Colors.RESET}")


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