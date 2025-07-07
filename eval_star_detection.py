#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pickle
from pathlib import Path
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from utils import get_celebrity_folder_path, get_celebrity_folder_name, get_image_files, get_average_embedding_path, load_pickle, get_env_float, print_error, print_summary, calculate_face_similarity, log
from utils_deepface import get_face_embeddings

# Load environment variables
load_dotenv()

def load_embedding(embedding_path):
    """Load the precomputed average embedding from pickle file."""
    embedding = load_pickle(embedding_path)
    if embedding is None:
        raise ValueError(f"Error loading embedding file: {embedding_path}")
    log(f"Loaded embedding with shape: {embedding.shape}")
    return embedding

def detect_and_compare_faces(image_path, reference_embedding, threshold=0.6):
    """
    Detect faces in image and compare with reference embedding.
    
    Returns:
        list: List of tuples (face_region, similarity_score) for matches above threshold
    """
    # Detect faces and get their embeddings
    face_analysis = get_face_embeddings(image_path)
    
    if not face_analysis:
        return []
        
    matches = []
    for i, face_data in enumerate(face_analysis):
        face_embedding = face_data['embedding']
        
        # Calculate cosine similarity
        similarity = calculate_face_similarity(face_embedding, reference_embedding)
        
        if similarity >= threshold:
            face_region = face_data['bounding_box']
            matches.append((face_region, similarity, i))
    
    return matches

def extract_face_crop(image_path, face_region, output_path):
    """Extract and save face crop from image."""
    try:
        # Read the original image
        img = cv2.imread(str(image_path))
        if img is None:
            return False
        
        # Extract face region
        x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
        face_crop = img[y:y+h, x:x+w]
        
        # Save the cropped face
        cv2.imwrite(str(output_path), face_crop)
        return True
        
    except Exception as e:
        print_error(f"Error extracting face crop: {e}")
        return False

def process_images(images_folder, embedding_path, threshold=0.6, output_folder="detected_headshots"):
    """
    Process all images in folder and detect matching faces.
    """
    images_folder = Path(images_folder)
    
    if not images_folder.exists():
        raise FileNotFoundError(f"Images folder not found: {images_folder}")
    
    # Load reference embedding
    reference_embedding = load_embedding(embedding_path)
    
    # Create output folder (clear existing files first)
    output_path = images_folder / output_folder
    
    # Clear existing files in output folder if it exists
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    
    output_path.mkdir(exist_ok=True)
    
    # Get all image files
    image_files = get_image_files(images_folder, exclude_subdirs=True)
    
    if not image_files:
        log(f"No image files found in {images_folder}")
        return
    
    log(f"Processing {len(image_files)} images...")
    
    total_detections = 0
    processed_images = 0
    
    for img_file in image_files:
        log(f"Processing: {img_file.name}")
        
        matches = detect_and_compare_faces(img_file, reference_embedding, threshold)
        
        if matches:
            processed_images += 1
            for face_region, similarity, face_idx in matches:
                # Create output filename
                base_name = img_file.stem
                output_filename = f"{base_name}_{similarity:.3f}.jpg"
                output_file_path = output_path / output_filename
                
                # Extract and save face crop
                if extract_face_crop(img_file, face_region, output_file_path):
                    log(f"  → Detected face with similarity {similarity:.3f} → {output_filename}")
                    total_detections += 1
                else:
                    log(f"  → Failed to extract face crop")
        else:
            log(f"  → No matching faces found")
    
    # Summary
    log(f"\nDetection Summary:")
    log(f"Images processed: {len(image_files)}")
    log(f"Images with detections: {processed_images}")
    log(f"Total faces detected: {total_detections}")
    log(f"Output folder: {output_path}")
    
    if total_detections == 0:
        print_error("No faces matching the reference were detected across all images.")
    else:
        print_summary(f"Face detection completed! Found {total_detections} matching faces across {processed_images} images.")

def main():
    parser = argparse.ArgumentParser(description='Detect star faces in images using precomputed embeddings')
    parser.add_argument('celebrity_name', help='Celebrity name (e.g., "Bill Murray")')
    # Get default threshold from environment variable
    default_threshold = get_env_float('TESTING_DETECTION_THRESHOLD', 0.6)
    parser.add_argument('--threshold', '-t', type=float, default=default_threshold,
                       help=f'Similarity threshold for face matching (default: {default_threshold})')
    parser.add_argument('--output', '-o', default='detected_headshots',
                       help='Output folder name (default: detected_headshots)')
    
    args = parser.parse_args()
    
    try:
        # Convert celebrity name to folder format (lowercase, spaces to underscores)
        celeb_folder = get_celebrity_folder_name(args.celebrity_name)
        
        # Construct paths automatically
        images_folder = get_celebrity_folder_path(args.celebrity_name, 'testing')
        embedding_file = get_average_embedding_path(args.celebrity_name, 'training')
        
        # Verify paths exist
        if not os.path.exists(images_folder):
            raise FileNotFoundError(f"Testing folder not found: {images_folder}")
        if not os.path.exists(embedding_file):
            raise FileNotFoundError(f"Embedding file not found: {embedding_file}")
        
        log(f"Using testing folder: {images_folder}")
        log(f"Using embedding file: {embedding_file}")
        
        process_images(images_folder, embedding_file, args.threshold, args.output)
        
    except Exception as e:
        print_error(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()