#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import pickle
import cv2
import shutil
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from utils import get_actor_folder_name, get_average_embedding_path, load_pickle, get_env_float, print_dry_run_header, print_dry_run_summary, print_error, print_summary, calculate_face_similarity, get_headshot_crop_coordinates, log

# Load environment variables
load_dotenv()

def load_actor_embedding(actor_name):
    """Load the precomputed average embedding for a actor."""
    embedding_path = get_average_embedding_path(actor_name, 'models')
    
    if not embedding_path.exists():
        raise FileNotFoundError(f"Actor embedding file not found: {embedding_path}")
    
    embedding = load_pickle(embedding_path)
    if embedding is None:
        raise ValueError(f"Error loading actor embedding from: {embedding_path}")
    
    log(f"Loaded actor embedding for '{actor_name}' with shape: {embedding.shape}")
    return embedding

def load_frame_face_data(pkl_path):
    """Load face data from a frame pickle file."""
    frame_data = load_pickle(pkl_path)
    if frame_data is None:
        print_error(f"Error loading face data from {pkl_path}")
    return frame_data

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
    
    log(f"Scanning {len(pkl_files)} frame files for face matches...")
    
    matches = []
    total_faces_scanned = 0
    
    for pkl_path in pkl_files:
        frame_data = load_frame_face_data(pkl_path)
        if frame_data is None:
            continue
        
        frame_file = frame_data.get('frame_file', pkl_path.stem + '.jpg')
        faces = [face for face in frame_data.get('faces', []) if face.get('isHeadshotable', True)]
        total_faces_scanned += len(faces)
        
        for face in faces:
            try:
                # Get face embedding and calculate similarity
                face_embedding = face['embedding']
                similarity = calculate_face_similarity(face_embedding, reference_embedding)
                
                if similarity >= threshold:
                    matches.append((similarity, frame_file, face, pkl_path))
                    
            except Exception as e:
                print_error(f"Error processing face in {pkl_path}: {e}")
                continue
    
    log(f"Scanned {total_faces_scanned} faces across {len(pkl_files)} frames")
    log(f"Found {len(matches)} faces above similarity threshold {threshold}")
    
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
        
        # Get crop coordinates using utility function
        img_width, img_height = img.shape[1], img.shape[0]
        crop_coords = get_headshot_crop_coordinates(bbox, img_width, img_height)
        x_start, y_start, x_end, y_end = crop_coords['x_start'], crop_coords['y_start'], crop_coords['x_end'], crop_coords['y_end']
        
        if crop_coords['clipped']:
            print_error(f"Face crop for {frame_file} is clipped, skipping extraction, this should have been prevented in 32_extract_frame_faces.py")
            return False

        face_crop = img[y_start:y_end, x_start:x_end]
        
        # Save the cropped face
        cv2.imwrite(str(output_path), face_crop)
        return True
        
    except Exception as e:
        print_error(f"Error extracting face crop: {e}")
        return False

def extract_top_headshots(actor_name, video_folder_path, threshold=0.6, dry_run=False):
    """
    Extract top 5 most similar headshots from video frames.
    """
    video_folder = Path(video_folder_path)
    
    if not video_folder.exists():
        raise FileNotFoundError(f"Video folder not found: {video_folder}")
    
    frames_dir = video_folder / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    
    # Create actor-specific headshots output directory
    actor_name_clean = get_actor_folder_name(actor_name)
    headshots_dir = video_folder / "headshots" / actor_name_clean
    
    if not dry_run:
        # Remove existing actor headshots folder and recreate it
        if headshots_dir.exists():
            shutil.rmtree(headshots_dir)
        headshots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load actor reference embedding
    reference_embedding = load_actor_embedding(actor_name)
    
    # Find all matching faces
    matches = calculate_face_similarities(frames_dir, reference_embedding, threshold)
    
    if not matches:
        print_error(f"No faces found matching '{actor_name}' above threshold {threshold}")
        return
    
    # Sort by similarity score (highest first) and take top 5
    matches.sort(key=lambda x: x[0], reverse=True)
    top_matches = matches[:5]
    
    log(f"\nTop {len(top_matches)} matches:")
    
    for i, (similarity, frame_file, face_data, pkl_path) in enumerate(top_matches, 1):
        # Extract frame number from filename (e.g., "00000001.jpg" -> "00000001")
        frame_position = Path(frame_file).stem
        
        # Create output filename with frame number
        output_filename = f"{actor_name_clean}_match_{similarity:.3f}_position_{frame_position}.jpg"
        output_path = headshots_dir / output_filename
        
        log(f"  {i}. {output_filename} (similarity: {similarity:.3f})")
        
        if dry_run:
            log(f"     Would extract from: {frame_file}")
            continue
        
        # Extract and save the face crop
        if extract_face_crop(frames_dir, frame_file, face_data, output_path):
            log(f"     → Saved to: {output_path}")
        else:
            print_error("Failed to extract face crop")
    
    if not dry_run:
        print_summary(f"Successfully extracted {len(top_matches)} headshots for {actor_name} to {headshots_dir}")
    else:
        print_summary(f"DRY RUN: Would extract {len(top_matches)} headshots for {actor_name}")

def main():
    parser = argparse.ArgumentParser(description='Extract top 5 most similar actor headshots from video frames')
    parser.add_argument('actor_name', help='Actor name (e.g., "Bill Murray")')
    parser.add_argument('video_folder_path', help='Path to video folder containing frames/ subdirectory')
    # Get default threshold from environment variable
    default_threshold = get_env_float('OPERATIONS_HEADSHOT_MATCH_THRESHOLD', 0.6)
    parser.add_argument('--threshold', '-t', type=float, default=default_threshold,
                       help=f'Similarity threshold for face matching (default: {default_threshold})')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be extracted without actually doing it')
    
    args = parser.parse_args()
    
    try:
        log(f"Extracting headshots for: {args.actor_name}")
        log(f"Video folder: {args.video_folder_path}")
        log(f"Similarity threshold: {args.threshold}")
        
        if args.dry_run:
            print_dry_run_header("No files will be created")
            log("")
        
        extract_top_headshots(
            args.actor_name,
            args.video_folder_path,
            args.threshold,
            args.dry_run
        )
        
        
    except Exception as e:
        print_error(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
