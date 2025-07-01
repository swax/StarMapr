#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pickle
from deepface import DeepFace
from pathlib import Path
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_embedding(embedding_path):
    """Load the precomputed average embedding from pickle file."""
    try:
        with open(embedding_path, 'rb') as f:
            embedding = pickle.load(f)
        print(f"Loaded embedding with shape: {embedding.shape}")
        return embedding
    except Exception as e:
        raise ValueError(f"Error loading embedding file: {e}")

def detect_and_compare_faces(image_path, reference_embedding, threshold=0.6):
    """
    Detect faces in image and compare with reference embedding.
    
    Returns:
        list: List of tuples (face_region, similarity_score) for matches above threshold
    """
    try:
        # Detect faces and get their embeddings
        face_analysis = DeepFace.represent(str(image_path), model_name='ArcFace', enforce_detection=False)
        
        if not face_analysis:
            return []
        
        matches = []
        for i, face_data in enumerate(face_analysis):
            face_embedding = np.array(face_data['embedding']).reshape(1, -1)
            reference_embedding_reshaped = reference_embedding.reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(face_embedding, reference_embedding_reshaped)[0][0]
            
            if similarity >= threshold:
                face_region = face_data['facial_area']
                matches.append((face_region, similarity, i))
        
        return matches
        
    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        return []

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
        print(f"Error extracting face crop: {e}")
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
    
    # Create output folder
    output_path = images_folder / output_folder
    output_path.mkdir(exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in images_folder.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {images_folder}")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    total_detections = 0
    processed_images = 0
    
    for img_file in image_files:
        print(f"Processing: {img_file.name}")
        
        matches = detect_and_compare_faces(img_file, reference_embedding, threshold)
        
        if matches:
            processed_images += 1
            for face_region, similarity, face_idx in matches:
                # Create output filename
                base_name = img_file.stem
                output_filename = f"{base_name}_face_{face_idx+1}_sim_{similarity:.3f}.jpg"
                output_file_path = output_path / output_filename
                
                # Extract and save face crop
                if extract_face_crop(img_file, face_region, output_file_path):
                    print(f"  → Detected face with similarity {similarity:.3f} → {output_filename}")
                    total_detections += 1
                else:
                    print(f"  → Failed to extract face crop")
        else:
            print(f"  → No matching faces found")
    
    # Summary
    print(f"\nDetection Summary:")
    print(f"Images processed: {len(image_files)}")
    print(f"Images with detections: {processed_images}")
    print(f"Total faces detected: {total_detections}")
    print(f"Output folder: {output_path}")
    
    if total_detections == 0:
        print("No faces matching the reference were detected across all images.")

def main():
    parser = argparse.ArgumentParser(description='Detect star faces in images using precomputed embeddings')
    parser.add_argument('celebrity_name', help='Celebrity name (e.g., "Bill Murray")')
    # Get default threshold from environment variable
    default_threshold = float(os.getenv('TESTING_DETECTION_THRESHOLD', 0.6))
    parser.add_argument('--threshold', '-t', type=float, default=default_threshold,
                       help=f'Similarity threshold for face matching (default: {default_threshold})')
    parser.add_argument('--output', '-o', default='detected_headshots',
                       help='Output folder name (default: detected_headshots)')
    
    args = parser.parse_args()
    
    try:
        # Convert celebrity name to folder format (lowercase, spaces to underscores)
        celeb_folder = args.celebrity_name.lower().replace(' ', '_')
        
        # Construct paths automatically
        images_folder = f"testing/{celeb_folder}/"
        embedding_file = f"training/{celeb_folder}/{celeb_folder}_average_embedding.pkl"
        
        # Verify paths exist
        if not os.path.exists(images_folder):
            raise FileNotFoundError(f"Testing folder not found: {images_folder}")
        if not os.path.exists(embedding_file):
            raise FileNotFoundError(f"Embedding file not found: {embedding_file}")
        
        print(f"Using testing folder: {images_folder}")
        print(f"Using embedding file: {embedding_file}")
        
        process_images(images_folder, embedding_file, args.threshold, args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()