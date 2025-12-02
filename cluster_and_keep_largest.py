#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter
from dotenv import load_dotenv
from utils import add_training_testing_args, get_mode_and_path_from_args, get_image_files, get_env_float, print_dry_run_header, print_dry_run_summary, print_error, print_summary, move_file_with_pkl, log
from utils_deepface import get_single_face_embedding

# Load environment variables
load_dotenv()


def cluster_and_keep_largest(actor_folder_path, eps=0.4, min_samples=2, dry_run=False):
    """
    Cluster all faces and keep only the largest cluster, moving others to outliers.

    Args:
        actor_folder_path (str): Path to actor folder containing images
        eps (float): DBSCAN epsilon parameter - maximum distance between samples (lower = stricter)
        min_samples (int): Minimum samples in a cluster
        dry_run (bool): If True, only report what would be moved without actually moving files
    """
    folder_path = Path(actor_folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Create outliers folder if it doesn't exist
    outliers_folder = folder_path / "outliers"

    # Get all image files (excluding those already in subfolders)
    image_files = [f for f in get_image_files(folder_path, exclude_subdirs=True)
                   if not f.name.startswith('.')]

    if len(image_files) < 3:
        log(f"Need at least 3 images to cluster. Found {len(image_files)} images.")
        return

    log(f"Clustering {len(image_files)} images...")
    if dry_run:
        print_dry_run_header("No files will be moved")
    log("")

    # Load embeddings from existing .pkl files
    embeddings = {}
    valid_images = []

    for img_file in image_files:
        log(f"Loading: {img_file.name}")
        embedding = get_single_face_embedding(img_file)
        if embedding is not None:
            embeddings[img_file] = embedding
            valid_images.append(img_file)
            log(f"  ✓ Embedding loaded")
        else:
            log(f"  ✗ No embedding found")

    if len(valid_images) < 3:
        log(f"\nInsufficient embeddings ({len(valid_images)}) to cluster.")
        return

    log(f"\nClustering {len(valid_images)} embeddings with DBSCAN...")
    log(f"Parameters: eps={eps}, min_samples={min_samples}")
    log("")

    # Create embedding matrix for clustering
    embedding_matrix = np.array([embeddings[img] for img in valid_images])

    # Run DBSCAN clustering
    # Note: DBSCAN works with distances, and cosine distance = 1 - cosine_similarity
    # Since embeddings are normalized, we can use euclidean distance which approximates cosine distance
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = clustering.fit_predict(embedding_matrix)

    # Map images to their cluster labels
    image_clusters = {img: label for img, label in zip(valid_images, labels)}

    # Count cluster sizes (label -1 means noise/outlier in DBSCAN)
    cluster_counts = Counter(labels)

    # Find the largest cluster (excluding noise label -1)
    valid_clusters = {label: count for label, count in cluster_counts.items() if label != -1}

    if not valid_clusters:
        log("No clusters found! All images classified as noise.")
        log("Try increasing 'eps' parameter or decreasing 'min_samples'.")
        return

    largest_cluster_label = max(valid_clusters, key=valid_clusters.get)
    largest_cluster_size = valid_clusters[largest_cluster_label]

    # Separate images into keep vs move
    keep_images = []
    move_images = []

    for img, label in image_clusters.items():
        if label == largest_cluster_label:
            keep_images.append(img)
        else:
            move_images.append((img, label))

    # Report results
    log(f"Clustering Results:")
    log(f"Total clusters found: {len(valid_clusters)}")
    log(f"Noise images (no cluster): {cluster_counts.get(-1, 0)}")
    log(f"Largest cluster: {largest_cluster_size} images (cluster {largest_cluster_label})")
    log("")

    log(f"Images to KEEP ({len(keep_images)}):")
    for img in keep_images:
        log(f"  ✓ {img.name}")
    log("")

    if move_images:
        log(f"Images to MOVE to outliers/ ({len(move_images)}):")
        for img, label in move_images:
            cluster_name = "noise" if label == -1 else f"cluster {label}"
            log(f"  → {img.name} ({cluster_name})")
        log("")

        # Move images if not in dry run mode
        if not dry_run:
            log(f"Moving {len(move_images)} images to {outliers_folder}...")

            total_moved_count = 0
            total_attempted_count = 0

            for img, label in move_images:
                moved_count, attempted_count = move_file_with_pkl(img, outliers_folder, dry_run)
                total_moved_count += moved_count
                total_attempted_count += attempted_count

            log(f"\nSuccessfully moved {total_moved_count} files")
        else:
            total_files_to_move = 0
            for img, label in move_images:
                moved_count, attempted_count = move_file_with_pkl(img, outliers_folder, dry_run=True)
                total_files_to_move += attempted_count
            print_dry_run_summary(total_files_to_move, "move to outliers folder")
    else:
        log("No images to move - all images are in the largest cluster!")

    # Summary
    if not dry_run:
        if move_images:
            print_summary(f"Clustering analysis completed. {len(keep_images)} clustered images. {len(move_images)} outliers.")
        else:
            print_summary("Clustering analysis completed. All images kept in largest cluster!")
    else:
        print_summary(f"DRY RUN: Would keep {len(keep_images)} images, move {len(move_images)} to outliers/")


def main():
    parser = argparse.ArgumentParser(description='Cluster faces and keep only the largest cluster')

    # Add standard training/testing arguments
    add_training_testing_args(parser)

    # Get default eps from environment variable (similar to threshold)
    default_eps = get_env_float('CLUSTERING_EPS', 0.4)
    parser.add_argument('--eps', type=float, default=default_eps,
                       help=f'DBSCAN epsilon parameter - max distance between samples (default: {default_eps})')
    parser.add_argument('--min-samples', type=int, default=2,
                       help='Minimum samples required to form a cluster (default: 2)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be moved without actually moving files')

    args = parser.parse_args()

    # Validate eps
    if not 0.0 <= args.eps <= 2.0:
        print_error("eps must be between 0.0 and 2.0")
        sys.exit(1)

    if args.min_samples < 1:
        print_error("min-samples must be at least 1")
        sys.exit(1)

    # Determine actor folder path based on arguments
    mode, actor_name, actor_folder_path = get_mode_and_path_from_args(args)

    try:
        cluster_and_keep_largest(actor_folder_path, args.eps, args.min_samples, args.dry_run)

    except Exception as e:
        print_error(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
