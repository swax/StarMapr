#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import shutil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from dotenv import load_dotenv
from utils import add_training_testing_args, get_mode_and_path_from_args, get_image_files, get_env_float, print_dry_run_header, print_dry_run_summary, print_error, print_summary, calculate_face_similarity, move_file_with_pkl, log
from utils_deepface import get_single_face_embedding

# Load environment variables
load_dotenv()


def find_face_outliers(celebrity_folder_path, similarity_threshold=0.1, dry_run=False):
    """
    Find and remove face outliers from celebrity training images.
    
    Compares all faces against each other using cosine similarity and identifies
    faces that are significantly different from the majority group.
    
    Args:
        celebrity_folder_path (str): Path to celebrity folder containing training images
        similarity_threshold (float): Minimum similarity to be considered same person (0.0-1.0)
        dry_run (bool): If True, only report what would be moved without actually moving files
    """
    folder_path = Path(celebrity_folder_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Create outliers folder if it doesn't exist (but only if we're not in dry run)
    outliers_folder = folder_path / "outliers"
    
    # Get all image files (excluding those already in subfolders)
    image_files = [f for f in get_image_files(folder_path, exclude_subdirs=True) 
                   if not f.name.startswith('.')]
    
    if len(image_files) < 3:
        log(f"Need at least 3 images to detect outliers. Found {len(image_files)} images.")
        return
    
    log(f"Analyzing {len(image_files)} images for face consistency...")
    if dry_run:
        print_dry_run_header("No files will be moved")
    log("")
    
    # Extract embeddings for all images
    embeddings = {}
    valid_images = []
    
    for img_file in image_files:
        log(f"Processing: {img_file.name}")
        embedding = get_single_face_embedding(img_file)
        if embedding is not None:
            embeddings[img_file] = embedding
            valid_images.append(img_file)
            log(f"  ✓ Face embedding extracted")
        else:
            log(f"  ✗ No face detected or processing error")
    
    if len(valid_images) < 3:
        log(f"\nInsufficient valid face embeddings ({len(valid_images)}) to detect outliers.")
        return
    
    log(f"\nComparing {len(valid_images)} face embeddings...")
    
    # Create similarity matrix
    embedding_matrix = np.array([embeddings[img] for img in valid_images])
    similarity_matrix = cosine_similarity(embedding_matrix)
    
    # For each image, calculate average similarity with all other images
    avg_similarities = {}
    for i, img in enumerate(valid_images):
        # Calculate average similarity excluding self-comparison
        similarities = similarity_matrix[i]
        avg_similarity = (similarities.sum() - similarities[i]) / (len(similarities) - 1)
        avg_similarities[img] = avg_similarity
    
    # Identify outliers based on similarity threshold
    outliers = []
    consensus_group = []
    
    for img, avg_sim in avg_similarities.items():
        if avg_sim < similarity_threshold:
            outliers.append((img, avg_sim))
        else:
            consensus_group.append((img, avg_sim))
    
    # Sort by similarity score for better reporting
    outliers.sort(key=lambda x: x[1])
    consensus_group.sort(key=lambda x: x[1], reverse=True)
    
    # Report results
    log(f"\nFace Consistency Analysis Results:")
    log(f"Similarity threshold: {similarity_threshold}")
    log(f"Images in consensus group: {len(consensus_group)}")
    log(f"Outlier images detected: {len(outliers)}")
    log("")
    
    if consensus_group:
        log("Consensus group (keeping):")
        for img, sim in consensus_group:
            log(f"  ✓ {img.name} (avg similarity: {sim:.3f})")
        log("")
    
    if outliers:
        log("Outlier images (will be moved):")
        for img, sim in outliers:
            log(f"  → {img.name} (avg similarity: {sim:.3f})")
        log("")
        
        # Move outlier images if not in dry run mode
        if not dry_run:
            log(f"Moving {len(outliers)} outlier images to {outliers_folder}...")
            
            total_moved_count = 0
            total_attempted_count = 0
            
            for img, sim in outliers:
                moved_count, attempted_count = move_file_with_pkl(img, outliers_folder, dry_run)
                total_moved_count += moved_count
                total_attempted_count += attempted_count
            
            log(f"\nSuccessfully moved {total_moved_count}/{total_attempted_count} files")
        else:
            total_files_to_move = 0
            for img, sim in outliers:
                moved_count, attempted_count = move_file_with_pkl(img, outliers_folder, dry_run=True)
                total_files_to_move += attempted_count
            print_dry_run_summary(total_files_to_move, "move outlier files to outliers folder")
    else:
        log("No outliers detected - all faces appear to be consistent!")
    
    # Summary statistics
    if valid_images:
        similarities = list(avg_similarities.values())
        log(f"\nSimilarity Statistics:")
        log(f"Mean similarity: {np.mean(similarities):.3f}")
        log(f"Min similarity: {np.min(similarities):.3f}")
        log(f"Max similarity: {np.max(similarities):.3f}")
        log(f"Std deviation: {np.std(similarities):.3f}")
    
    # Add success summary
    if not dry_run:
        if outliers:
            print_summary(f"Face outlier analysis completed. Moved {len(outliers)} outlier images to outliers/ folder.")
        else:
            print_summary("Face outlier analysis completed. All faces appear to be consistent!")
    else:
        print_summary(f"DRY RUN: Face outlier analysis completed. Would move {len(outliers)} outlier images.")

def main():
    parser = argparse.ArgumentParser(description='Remove face outliers from celebrity training images')
    
    # Add standard training/testing arguments
    add_training_testing_args(parser)
    
    # Get default threshold from environment variable
    default_threshold = get_env_float('TRAINING_OUTLIER_THRESHOLD', 0.1)
    parser.add_argument('--threshold', type=float, default=default_threshold,
                       help=f'Similarity threshold for outlier detection (default: {default_threshold})')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be moved without actually moving files')
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print_error("Threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Determine celebrity folder path based on arguments
    mode, celebrity_name, celebrity_folder_path = get_mode_and_path_from_args(args)
    
    try:
        find_face_outliers(celebrity_folder_path, args.threshold, args.dry_run)
        
    except Exception as e:
        print_error(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()