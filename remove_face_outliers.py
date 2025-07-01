#!/usr/bin/env python3
import os
import sys
import argparse
from deepface import DeepFace
from pathlib import Path
import shutil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def get_face_embedding(image_path):
    """
    Get face embedding for an image using DeepFace ArcFace model.
    
    Args:
        image_path (Path): Path to the image file
        
    Returns:
        np.array or None: Face embedding vector, None if error or no face detected
    """
    try:
        # Use DeepFace to get face embedding (enforce_detection=False to avoid exceptions)
        embeddings = DeepFace.represent(str(image_path), model_name='ArcFace', enforce_detection=False)
        if embeddings and len(embeddings) > 0:
            # Return the first face embedding if multiple faces detected
            return np.array(embeddings[0]['embedding'])
        return None
    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        return None

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
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in folder_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions 
                   and not f.name.startswith('.')]
    
    if len(image_files) < 3:
        print(f"Need at least 3 images to detect outliers. Found {len(image_files)} images.")
        return
    
    print(f"Analyzing {len(image_files)} images for face consistency...")
    if dry_run:
        print("DRY RUN MODE - No files will be moved")
    print()
    
    # Extract embeddings for all images
    embeddings = {}
    valid_images = []
    
    for img_file in image_files:
        print(f"Processing: {img_file.name}")
        embedding = get_face_embedding(img_file)
        if embedding is not None:
            embeddings[img_file] = embedding
            valid_images.append(img_file)
            print(f"  ✓ Face embedding extracted")
        else:
            print(f"  ✗ No face detected or processing error")
    
    if len(valid_images) < 3:
        print(f"\nInsufficient valid face embeddings ({len(valid_images)}) to detect outliers.")
        return
    
    print(f"\nComparing {len(valid_images)} face embeddings...")
    
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
    print(f"\nFace Consistency Analysis Results:")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Images in consensus group: {len(consensus_group)}")
    print(f"Outlier images detected: {len(outliers)}")
    print()
    
    if consensus_group:
        print("Consensus group (keeping):")
        for img, sim in consensus_group:
            print(f"  ✓ {img.name} (avg similarity: {sim:.3f})")
        print()
    
    if outliers:
        print("Outlier images (will be moved):")
        for img, sim in outliers:
            print(f"  → {img.name} (avg similarity: {sim:.3f})")
        print()
        
        # Move outlier images if not in dry run mode
        if not dry_run:
            outliers_folder.mkdir(exist_ok=True)
            print(f"Moving {len(outliers)} outlier images to {outliers_folder}...")
            
            moved_count = 0
            for img, sim in outliers:
                try:
                    destination = outliers_folder / img.name
                    shutil.move(str(img), str(destination))
                    print(f"  ✓ Moved: {img.name}")
                    moved_count += 1
                except Exception as e:
                    print(f"  ✗ Failed to move {img.name}: {e}")
            
            print(f"\nSuccessfully moved {moved_count}/{len(outliers)} outlier images")
        else:
            print(f"DRY RUN: Would move {len(outliers)} outlier images to outliers folder")
    else:
        print("No outliers detected - all faces appear to be consistent!")
    
    # Summary statistics
    if valid_images:
        similarities = list(avg_similarities.values())
        print(f"\nSimilarity Statistics:")
        print(f"Mean similarity: {np.mean(similarities):.3f}")
        print(f"Min similarity: {np.min(similarities):.3f}")
        print(f"Max similarity: {np.max(similarities):.3f}")
        print(f"Std deviation: {np.std(similarities):.3f}")

def main():
    parser = argparse.ArgumentParser(description='Remove face outliers from celebrity training images')
    
    # Mutually exclusive group for mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--training', metavar='CELEBRITY_NAME',
                           help='Remove face outliers from training/CELEBRITY_NAME/ folder')
    mode_group.add_argument('--testing', metavar='CELEBRITY_NAME', 
                           help='Remove face outliers from testing/CELEBRITY_NAME/ folder')
    
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Similarity threshold for outlier detection (default: 0.1)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be moved without actually moving files')
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print("Error: Threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Determine celebrity folder path based on arguments
    if args.training:
        celebrity_name = args.training.lower().replace(' ', '_')
        celebrity_folder_path = f'training/{celebrity_name}/'
    else:  # args.testing
        celebrity_name = args.testing.lower().replace(' ', '_')
        celebrity_folder_path = f'testing/{celebrity_name}/'
    
    try:
        find_face_outliers(celebrity_folder_path, args.threshold, args.dry_run)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()