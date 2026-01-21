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
from utils import get_average_embedding_path, load_pickle, get_env_float, print_dry_run_header, print_dry_run_summary, print_error, print_summary, calculate_face_similarity, get_headshot_crop_coordinates, log

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

def get_actors_from_headshots_dir(video_folder_path):
    """Get list of actors from headshots directory folder names."""
    video_folder = Path(video_folder_path)
    headshots_dir = video_folder / "headshots"
    
    if not headshots_dir.exists():
        return []
    
    # Get all subdirectories in headshots folder
    actor_folders = [f.name for f in headshots_dir.iterdir() if f.is_dir()]
    
    # Convert folder names back to actor names (replace _ with space and title case)
    actors = []
    for folder_name in actor_folders:
        actor_name = folder_name.replace('_', ' ').title()
        actors.append(actor_name)
    
    return actors

def calculate_frame_score(frames_dir, frame_file, actor_embeddings, threshold=0.4):
    """
    Calculate average match score for all actors in a single frame.
    
    Returns:
        tuple: (average_score, actor_matches_dict) or (0.0, {}) if no matches
    """
    # Find the corresponding pickle file
    pkl_path = frames_dir / (Path(frame_file).stem + '.pkl')
    
    if not pkl_path.exists():
        return 0.0, {}
    
    frame_data = load_frame_face_data(pkl_path)
    if frame_data is None:
        return 0.0, {}
    
    faces = frame_data.get('faces', [])
    if not faces:
        return 0.0, {}
    
    # Track best match for each actor
    actor_matches = {}
    
    for actor_name, reference_embedding in actor_embeddings.items():
        best_similarity = 0.0
        best_face = None
        
        for face in faces:
            try:
                face_embedding = face['embedding']
                similarity = calculate_face_similarity(face_embedding, reference_embedding)
                
                if similarity >= threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_face = face
            except Exception as e:
                continue
        
        if best_similarity > 0:
            actor_matches[actor_name] = {
                'similarity': best_similarity,
                'face': best_face
            }
    
    # Calculate weighted score: count of actors + average similarity quality
    if not actor_matches:
        return 0.0, {}
    
    actor_count = len(actor_matches)
    average_similarity = sum(match['similarity'] for match in actor_matches.values()) / len(actor_matches)
    # Weight: more actors found is better, with similarity as tiebreaker
    weighted_score = actor_count + (average_similarity * 0.1)  # Small weight for similarity
    return weighted_score, actor_matches

def extract_middle_frame_thumbnail(video_folder_path, frames_dir):
    """Extract the middle frame as a fallback thumbnail."""
    try:
        # Get all frame files
        frame_files = sorted([f for f in frames_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        if not frame_files:
            return False
        
        # Get middle frame
        middle_index = len(frame_files) // 2
        middle_frame = frame_files[middle_index]
        
        # Copy to thumbnail1.jpg in video directory
        thumbnail_path = Path(video_folder_path) / "thumbnail1.jpg"
        shutil.copy2(middle_frame, thumbnail_path)
        
        log(f"Extracted middle frame as fallback: {thumbnail_path}")
        return True
        
    except Exception as e:
        print_error(f"Error extracting middle frame: {e}")
        return False

def extract_video_thumbnails(video_folder_path, threshold=0.4, dry_run=False):
    """
    Extract top 3 thumbnails from video frames based on highest average actor match scores.
    """
    video_folder = Path(video_folder_path)
    
    if not video_folder.exists():
        raise FileNotFoundError(f"Video folder not found: {video_folder}")
    
    frames_dir = video_folder / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    
    # Get actors from headshots directory
    actors = get_actors_from_headshots_dir(video_folder_path)
    
    if not actors:
        log("Warning: No actor folders found in headshots directory")
        log("Using middle frame as thumbnail")
        
        if not dry_run:
            extract_middle_frame_thumbnail(video_folder_path, frames_dir)
        else:
            log("DRY RUN: Would extract middle frame as thumbnail1.jpg")
        return
    
    log(f"Found {len(actors)} actors: {', '.join(actors)}")
    
    # Load embeddings for all actors
    actor_embeddings = {}
    for actor_name in actors:
        try:
            embedding = load_actor_embedding(actor_name)
            actor_embeddings[actor_name] = embedding
        except Exception as e:
            print_error(f"Could not load embedding for {actor_name}: {e}")
            continue
    
    if not actor_embeddings:
        print_error("No actor embeddings could be loaded")
        return
    
    log(f"Loaded embeddings for {len(actor_embeddings)} actors")
    
    # Get all frame files
    frame_files = sorted([f.name for f in frames_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if not frame_files:
        raise ValueError(f"No frame images found in {frames_dir}")
    
    log(f"Scanning {len(frame_files)} frames for multi-actor matches...")
    
    # Score all frames
    frame_scores = []
    for frame_file in frame_files:
        average_score, actor_matches = calculate_frame_score(frames_dir, frame_file, actor_embeddings, threshold)
        
        if average_score > 0:
            frame_scores.append((average_score, frame_file, actor_matches))
    
    if not frame_scores:
        log("Warning: No frames found with actor matches above threshold")
        log("Using middle frame as thumbnail")
        
        if not dry_run:
            extract_middle_frame_thumbnail(video_folder_path, frames_dir)
        else:
            log("DRY RUN: Would extract middle frame as thumbnail1.jpg")
        return
    
    # Sort by average score (highest first) and take top 3
    frame_scores.sort(key=lambda x: x[0], reverse=True)
    top_frames = frame_scores[:3]
    
    log(f"\nTop {len(top_frames)} thumbnail candidates:")
    
    for i, (average_score, frame_file, actor_matches) in enumerate(top_frames, 1):
        thumbnail_filename = f"thumbnail{i}.jpg"
        thumbnail_path = video_folder / thumbnail_filename
        
        log(f"  {i}. {thumbnail_filename} (avg score: {average_score:.3f})")
        log(f"     Frame: {frame_file}")
        
        # Show which actors were found
        found_actors = list(actor_matches.keys())
        log(f"     Actors found: {', '.join(found_actors)} ({len(found_actors)}/{len(actors)})")
        
        for actor_name, match_info in actor_matches.items():
            log(f"       {actor_name}: {match_info['similarity']:.3f}")
        
        if dry_run:
            log(f"     Would copy to: {thumbnail_path}")
            continue
        
        # Copy frame to thumbnail
        try:
            frame_path = frames_dir / frame_file
            shutil.copy2(frame_path, thumbnail_path)
            log(f"     → Saved to: {thumbnail_path}")
        except Exception as e:
            print_error(f"Error copying thumbnail: {e}")
    
    if not dry_run:
        print_summary(f"Successfully extracted {len(top_frames)} thumbnails to {video_folder}")
    else:
        print_summary(f"DRY RUN: Would extract {len(top_frames)} thumbnails")

def main():
    parser = argparse.ArgumentParser(description='Extract top 3 video thumbnails based on highest average actor match scores')
    parser.add_argument('video_folder_path', help='Path to video folder containing frames/ and headshots/ subdirectories')
    # Get default threshold from environment variable
    default_threshold = get_env_float('OPERATIONS_HEADSHOT_MATCH_THRESHOLD', 0.4)
    parser.add_argument('--threshold', '-t', type=float, default=default_threshold,
                       help=f'Similarity threshold for face matching (default: {default_threshold})')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be extracted without actually doing it')
    
    args = parser.parse_args()
    
    try:
        log(f"Extracting thumbnails from: {args.video_folder_path}")
        log(f"Similarity threshold: {args.threshold}")
        
        if args.dry_run:
            print_dry_run_header("No files will be created")
            log("")
        
        extract_video_thumbnails(
            args.video_folder_path,
            args.threshold,
            args.dry_run
        )
        
        
    except Exception as e:
        print_error(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
