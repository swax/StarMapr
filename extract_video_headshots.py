#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import pickle
import cv2
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from utilities import print_error, print_summary

# Load environment variables
load_dotenv()

def load_celebrity_embedding(celebrity_name):
    """Load the precomputed average embedding for a celebrity."""
    try:
        # Convert celebrity name to folder format
        celeb_folder = celebrity_name.lower().replace(' ', '_')
        embedding_path = Path(f"training/{celeb_folder}/{celeb_folder}_average_embedding.pkl")
        
        if not embedding_path.exists():
            raise FileNotFoundError(f"Celebrity embedding file not found: {embedding_path}")
        
        with open(embedding_path, 'rb') as f:
            embedding = pickle.load(f)
        
        print(f"Loaded celebrity embedding for '{celebrity_name}' with shape: {embedding.shape}")
        return embedding
        
    except Exception as e:
        raise ValueError(f"Error loading celebrity embedding: {e}")

def load_frame_face_data(pkl_path):
    """Load face data from a frame pickle file."""
    try:
        with open(pkl_path, 'rb') as f:
            frame_data = pickle.load(f)
        return frame_data
    except Exception as e:
        print_error(f"Error loading face data from {pkl_path}: {e}")
        return None

def calculate_face_similarities(frames_dir, reference_embedding, threshold=0.6):
    """
    Scan all pickle files in frames directory and calculate similarities.
    
    Returns:
        list: List of tuples (similarity_score, frame_file, face_data, pkl_path)
    """
    frames_dir = Path(frames_dir)
    
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    
    # Get all pickle files
    pkl_files = list(frames_dir.glob("*.pkl"))
    
    if not pkl_files:
        raise ValueError(f"No pickle files found in {frames_dir}")
    
    print(f"Scanning {len(pkl_files)} frame files for face matches...")
    
    matches = []
    total_faces_scanned = 0
    
    for pkl_path in pkl_files:
        frame_data = load_frame_face_data(pkl_path)
        if frame_data is None:
            continue
        
        frame_file = frame_data.get('frame_file', pkl_path.stem + '.jpg')
        faces = frame_data.get('faces', [])
        total_faces_scanned += len(faces)
        
        for face in faces:
            try:
                # Get face embedding
                face_embedding = np.array(face['embedding']).reshape(1, -1)
                reference_embedding_reshaped = reference_embedding.reshape(1, -1)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(face_embedding, reference_embedding_reshaped)[0][0]
                
                if similarity >= threshold:
                    matches.append((similarity, frame_file, face, pkl_path))
                    
            except Exception as e:
                print_error(f"Error processing face in {pkl_path}: {e}")
                continue
    
    print(f"Scanned {total_faces_scanned} faces across {len(pkl_files)} frames")
    print(f"Found {len(matches)} faces above similarity threshold {threshold}")
    
    return matches

def extract_face_crop(frames_dir, frame_file, face_data, output_path):
    """Extract and save face crop from frame image."""
    try:
        # Find the frame image file
        frame_path = Path(frames_dir) / frame_file
        
        if not frame_path.exists():
            print_error(f"Frame file not found: {frame_path}")
            return False
        
        # Read the frame image
        img = cv2.imread(str(frame_path))
        if img is None:
            print_error(f"Could not read image: {frame_path}")
            return False
        
        # Extract face region using bounding box
        bbox = face_data['bounding_box']
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        
        # Add some padding around the face (10% on each side)
        padding = int(min(w, h) * 0.1)
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img.shape[1], x + w + padding)
        y_end = min(img.shape[0], y + h + padding)
        
        face_crop = img[y_start:y_end, x_start:x_end]
        
        # Save the cropped face
        cv2.imwrite(str(output_path), face_crop)
        return True
        
    except Exception as e:
        print_error(f"Error extracting face crop: {e}")
        return False

def extract_top_headshots(celebrity_name, video_folder_path, threshold=0.6, dry_run=False):
    """
    Extract top 5 most similar headshots from video frames.
    """
    video_folder = Path(video_folder_path)
    
    if not video_folder.exists():
        raise FileNotFoundError(f"Video folder not found: {video_folder}")
    
    frames_dir = video_folder / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    
    # Create headshots output directory
    headshots_dir = video_folder / "headshots"
    
    if not dry_run:
        headshots_dir.mkdir(exist_ok=True)
        
        # Clean up existing headshots
        for existing_file in headshots_dir.glob("*.jpg"):
            existing_file.unlink()
    
    # Load celebrity reference embedding
    reference_embedding = load_celebrity_embedding(celebrity_name)
    
    # Find all matching faces
    matches = calculate_face_similarities(frames_dir, reference_embedding, threshold)
    
    if not matches:
        print_error(f"No faces found matching '{celebrity_name}' above threshold {threshold}")
        return
    
    # Sort by similarity score (highest first) and take top 5
    matches.sort(key=lambda x: x[0], reverse=True)
    top_matches = matches[:5]
    
    print(f"\nTop {len(top_matches)} matches:")
    
    celeb_name_clean = celebrity_name.lower().replace(' ', '_')
    
    for i, (similarity, frame_file, face_data, pkl_path) in enumerate(top_matches, 1):
        bbox = face_data['bounding_box']
        width, height = bbox['w'], bbox['h']
        
        # Create output filename with width x height
        output_filename = f"{celeb_name_clean}_{similarity:.3f}_{width}x{height}.jpg"
        output_path = headshots_dir / output_filename
        
        print(f"  {i}. {output_filename} (similarity: {similarity:.3f})")
        
        if dry_run:
            print(f"     Would extract from: {frame_file}")
            continue
        
        # Extract and save the face crop
        if extract_face_crop(frames_dir, frame_file, face_data, output_path):
            print(f"     → Saved to: {output_path}")
        else:
            print_error("Failed to extract face crop")
    
    if not dry_run:
        print_summary(f"Successfully extracted {len(top_matches)} headshots for {celebrity_name} to {headshots_dir}")
    else:
        print_summary(f"DRY RUN: Would extract {len(top_matches)} headshots for {celebrity_name}")

def main():
    parser = argparse.ArgumentParser(description='Extract top 5 most similar celebrity headshots from video frames')
    parser.add_argument('celebrity_name', help='Celebrity name (e.g., "Bill Murray")')
    parser.add_argument('video_folder_path', help='Path to video folder containing frames/ subdirectory')
    # Get default threshold from environment variable
    default_threshold = float(os.getenv('OPERATIONS_HEADSHOT_MATCH_THRESHOLD', 0.6))
    parser.add_argument('--threshold', '-t', type=float, default=default_threshold,
                       help=f'Similarity threshold for face matching (default: {default_threshold})')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be extracted without actually doing it')
    
    args = parser.parse_args()
    
    try:
        print(f"Extracting headshots for: {args.celebrity_name}")
        print(f"Video folder: {args.video_folder_path}")
        print(f"Similarity threshold: {args.threshold}")
        
        if args.dry_run:
            print("DRY RUN - No files will be created\n")
        
        extract_top_headshots(
            args.celebrity_name,
            args.video_folder_path,
            args.threshold,
            args.dry_run
        )
        
        
    except Exception as e:
        print_error(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()