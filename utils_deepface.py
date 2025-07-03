"""
StarMapr DeepFace Utilities Module

DeepFace-specific utility functions that require DeepFace and related AI dependencies.
This module is separated from utils.py to minimize CUDA warnings for scripts
that don't need DeepFace functionality.
"""

import numpy as np
import pickle
from pathlib import Path
from utils import print_error, get_env_int

def get_face_embeddings(image_path, enforce_detection=False):
    """
    Get face embeddings using DeepFace ArcFace model with pkl file caching.
    
    This function automatically creates/reads .pkl files alongside image files
    containing face analysis data (embeddings + bounding boxes). If a .pkl file
    already exists, it loads and returns the cached data instead of reprocessing.
    
    Args:
        image_path (str or Path): Path to the image
        enforce_detection (bool): Whether to enforce face detection
        
    Returns:
        list: List of face analysis results with structured data, empty list if error
    """
    image_path = Path(image_path)
    pkl_path = image_path.with_suffix('.pkl')
    
    # Check if cached data exists
    if pkl_path.exists():
        try:
            with open(pkl_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Return the faces data from the cached structure
            return cached_data.get('faces', [])
        except Exception as e:
            print_error(f"Error loading cached face data {pkl_path.name}: {e}")
            # Fall through to regenerate
    
    # Generate face embeddings
    try:
        # Delay the import to avoid the mess of unavoidable CUDA warnings that come with it
        from deepface import DeepFace
        face_analysis = DeepFace.represent(
            str(image_path), 
            model_name='ArcFace', 
            enforce_detection=enforce_detection
        )
        
        if not face_analysis:
            return []
        
        # Convert to structured format with face IDs and bounding boxes
        # Filter out faces smaller than minimum size
        min_face_size = get_env_int('MIN_FACE_SIZE', 50)
        faces_data = []
        
        # Load image for face extraction
        try:
            import cv2
            image = cv2.imread(str(image_path))
            image_height, image_width = image.shape[:2]
            faces_dir = image_path.parent / 'faces'
            faces_dir.mkdir(exist_ok=True)
            base_name = image_path.stem
        except Exception as e:
            print_error(f"Error setting up face extraction for {image_path.name}: {e}")
            image = None
            image_height, image_width = 0, 0
        
        for i, face_data in enumerate(face_analysis):
            face_region = face_data['facial_area']
            face_width = face_region['w']
            face_height = face_region['h']
            
            # Skip faces smaller than minimum size
            if face_width < min_face_size or face_height < min_face_size:
                continue
            
            # Skip faces that are nearly the same size as the image (within 3px) Sometimes it thinks the whole image is a face
            if (abs(face_width - image_width) <= 3 and abs(face_height - image_height) <= 3):
                continue
            
            # Extract and save face crop
            if image is not None:
                try:
                    x, y, w, h = face_region['x'], face_region['y'], face_width, face_height
                    face_crop = image[y:y+h, x:x+w]
                    face_filename = f"{base_name}_face_{len(faces_data)+1}{image_path.suffix}"
                    face_path = faces_dir / face_filename
                    cv2.imwrite(str(face_path), face_crop)
                except Exception as e:
                    print_error(f"Error saving face crop {i+1} for {image_path.name}: {e}")
                
            faces_data.append({
                'face_id': len(faces_data) + 1,
                'bounding_box': {
                    'x': face_region['x'],
                    'y': face_region['y'],
                    'w': face_width,
                    'h': face_height
                },
                'embedding': face_data['embedding']
            })
        
        # Cache the structured data
        try:
            frame_data = {
                'frame_file': image_path.name,
                'total_faces': len(faces_data),
                'faces': faces_data
            }
            
            with open(pkl_path, 'wb') as f:
                pickle.dump(frame_data, f)
        except Exception as e:
            print_error(f"Error caching face data to {pkl_path.name}: {e}")
        
        return faces_data
        
    except Exception as e:
        print_error(f"Error processing {image_path.name}: {e}")
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



