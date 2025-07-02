"""
StarMapr DeepFace Utilities Module

DeepFace-specific utility functions that require DeepFace and related AI dependencies.
This module is separated from utils.py to minimize CUDA warnings for scripts
that don't need DeepFace functionality.
"""

import numpy as np
from pathlib import Path
from deepface import DeepFace
from utils import print_error


def get_face_embeddings(image_path, enforce_detection=False):
    """
    Get face embeddings using DeepFace ArcFace model.
    
    Args:
        image_path (str or Path): Path to the image
        enforce_detection (bool): Whether to enforce face detection
        
    Returns:
        list: List of face analysis results, empty list if error
    """
    try:
        face_analysis = DeepFace.represent(
            str(image_path), 
            model_name='ArcFace', 
            enforce_detection=enforce_detection
        )
        return face_analysis if face_analysis else []
    except Exception as e:
        print_error(f"Error processing {Path(image_path).name}: {e}")
        return []


def get_single_face_embedding(image_path):
    """
    Get embedding for single face image.
    
    Args:
        image_path (str or Path): Path to the image
        
    Returns:
        numpy.ndarray or None: Face embedding array, None if error/no face
    """
    embeddings = get_face_embeddings(image_path)
    return np.array(embeddings[0]['embedding']) if embeddings else None


