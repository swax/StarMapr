#!/usr/bin/env python3
"""
StarMapr Training Pipeline Script

Runs the complete training pipeline for an actor:
- Downloads training images iteratively until minimum threshold met
- Removes duplicates, bad images, and outliers
- Generates average embeddings

Usage:
    python3 run_training_pipeline.py "Actor Name" "Show Name"
    python3 run_training_pipeline.py "Actor Name" "Show Name" --max-pages 5 --min-images 20
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path
from dotenv import load_dotenv
from utils import (
    get_actor_folder_path, get_image_files, get_env_int,
    get_average_embedding_path, print_error, ensure_folder_exists, get_venv_python
)

# Load environment variables
load_dotenv()

def print_header(text):
    """Print a header in blue color."""
    blue = '\033[94m'
    reset = '\033[0m'
    print(f"{blue}{text}{reset}")


def fatal_error(message):
    print_error(f"❌ {message}")
    sys.exit(1)


def restore_outliers_to_training(actor_name, mode='training'):
    """
    Move images from outliers folder back to training/testing folder.

    Args:
        actor_name (str): Name of the actor
        mode (str): 'training' or 'testing'

    Returns:
        int: Number of files moved back
    """
    folder_path = get_actor_folder_path(actor_name, mode)
    outliers_folder = Path(folder_path) / 'outliers'

    if not outliers_folder.exists():
        return 0

    # Get image files from outliers folder
    outlier_images = get_image_files(outliers_folder)
    if not outlier_images:
        return 0

    files_moved = 0
    for img_file in outlier_images:
        # Move image file
        dest_path = Path(folder_path) / img_file.name
        shutil.move(str(img_file), str(dest_path))
        files_moved += 1

        # Also move the corresponding .pkl file if it exists
        pkl_file = img_file.with_suffix('.pkl')
        if pkl_file.exists():
            pkl_dest = Path(folder_path) / pkl_file.name
            shutil.move(str(pkl_file), str(pkl_dest))

    return files_moved


def save_best_group(training_folder, image_files):
    """
    Save the list of best group filenames to a text file.

    Args:
        training_folder (str): Path to training folder
        image_files (list): List of Path objects representing image files
    """
    best_group_folder = Path(training_folder) / 'best_group'
    ensure_folder_exists(best_group_folder)
    best_group_file = best_group_folder / 'best_group.txt'

    # Extract just the filenames (not full paths)
    filenames = [img.name for img in image_files]

    # Save to text file, one filename per line
    with open(best_group_file, 'w') as f:
        f.write('\n'.join(filenames))

    print(f"✓ Saved best group of {len(filenames)} images")


def restore_best_group(actor_name, mode='training'):
    """
    Move files not in the best group text file to the outliers folder.

    Args:
        actor_name (str): Name of the actor
        mode (str): 'training' or 'testing'

    Returns:
        int: Number of files moved to outliers
    """
    # Final restoration of outliers before generating embeddings
    restore_outliers_to_training(actor_name, 'training')

    folder_path = get_actor_folder_path(actor_name, mode)
    best_group_file = Path(folder_path) / 'best_group' / 'best_group.txt'

    # If no best group file exists, nothing to restore
    if not best_group_file.exists():
        print("No best group file found, skipping restoration")
        return 0

    # Read the best group filenames
    with open(best_group_file, 'r') as f:
        best_filenames = set(line.strip() for line in f if line.strip())

    # Get current image files in the folder
    current_files = get_image_files(folder_path)

    # Create outliers folder if needed
    outliers_folder = Path(folder_path) / 'outliers'
    ensure_folder_exists(outliers_folder)

    files_moved = 0
    for img_file in current_files:
        if img_file.name not in best_filenames:
            dest_path = outliers_folder / img_file.name
            shutil.move(str(img_file), str(dest_path))
            files_moved += 1

            # Also move the corresponding .pkl file if it exists
            pkl_file = img_file.with_suffix('.pkl')
            if pkl_file.exists():
                pkl_dest = outliers_folder / pkl_file.name
                shutil.move(str(pkl_file), str(pkl_dest))

    if files_moved > 0:
        print(f"✓ Best group had {len(best_filenames)} images. {files_moved} outliers.")

    return files_moved


def run_subprocess_command(command_list, description):
    """
    Run a subprocess command with error handling.

    Args:
        command_list (list): Command and arguments to run
        description (str): Description of the command for error reporting

    Returns:
        bool: True if successful, False if failed
    """
    try:
        print(f"Running: {description}")
        # Don't live stream the output because it shows unavoidable cuda errors that fills the context
        result = subprocess.run(command_list, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed: {description}")
        return False


def check_image_threshold(training_folder, min_images, best_image_count):
    """
    Check if training folder has reached minimum image threshold.

    Args:
        training_folder (str): Path to training folder
        min_images (int): Minimum images required
        best_image_count (int): Best count achieved so far

    Returns:
        tuple: (threshold_met: bool, current_count: int, updated_best_count: int)
    """
    current_images = get_image_files(training_folder)
    image_count = len(current_images)

    # Update best count if current is higher
    if image_count > best_image_count:
        best_image_count = image_count
        save_best_group(training_folder, current_images)

    # Check if threshold is met
    threshold_met = image_count >= min_images
    if threshold_met:
        print(f"✓ Achieved minimum training images ({image_count} >= {min_images})")

    return threshold_met, image_count, best_image_count


def run_training_pipeline(actor_name, show_name, max_pages, min_images):
    """
    Run the training pipeline until minimum images achieved or max pages reached.

    Args:
        actor_name (str): Name of the actor
        show_name (str): Name of the show
        max_pages (int): Maximum pages to download
        min_images (int): Minimum training images required

    Returns:
        tuple: (success: bool, final_image_count: int)
    """
    print_header(f"\n=== TRAINING PIPELINE for '{actor_name}' ===")

    training_folder = get_actor_folder_path(actor_name, 'training')
    ensure_folder_exists(training_folder)

    best_image_count = 0

    for page in range(1, max_pages + 1):
        print_header(f"\n--- Training Page {page} ---")

        # Step 1: Download training images
        download_cmd = [
            get_venv_python(), 'download_actor_images.py', actor_name,
            '--training', '--show', show_name, '--page', str(page)
        ]
        if not run_subprocess_command(download_cmd, f"Downloading training images (page {page})"):
            fatal_error(f"Failed to download training images for page {page}")

        # Step 2: Remove duplicates
        dedup_cmd = [get_venv_python(), 'remove_dupe_training_images.py', '--training', actor_name]
        if not run_subprocess_command(dedup_cmd, "Removing duplicate images"):
            fatal_error("Failed to remove duplicate images")

        # Step 3: Remove bad face counts (not exactly 1 face)
        bad_cmd = [get_venv_python(), 'remove_bad_training_images.py', '--training', actor_name]
        if not run_subprocess_command(bad_cmd, "Removing bad training images"):
            fatal_error("Failed to remove bad training images")

        current_images = get_image_files(training_folder)
        image_count = len(current_images)
        if image_count == 0:
            print_header("No images left after cleaning, continuing to next page...")
            continue

        # Step 4a: Try similarity-based outlier detection first (works better with fewer images)
        restore_outliers_to_training(actor_name, 'training')

        outlier_cmd = [get_venv_python(), 'remove_face_outliers.py', '--training', actor_name]
        if not run_subprocess_command(outlier_cmd, "Removing face outliers (similarity based)"):
            fatal_error("Failed to remove face outliers")

        threshold_met, image_count, best_image_count = check_image_threshold(
            training_folder, min_images, best_image_count
        )
        if threshold_met:
            break

        # Step 4b: Try DBSCAN clustering, works better with lots of images where there might be an island of good ones that all match
        restore_outliers_to_training(actor_name, 'training')

        cluster_cmd = [get_venv_python(), 'cluster_and_keep_largest.py', '--training', actor_name]
        if not run_subprocess_command(cluster_cmd, "Removing face outliers (clustering based)"):
            fatal_error("Failed to cluster faces")

        threshold_met, image_count, best_image_count = check_image_threshold(
            training_folder, min_images, best_image_count
        )
        if threshold_met:
            break

        # Still not enough images, continue to next page
        if page < max_pages:
            print(f"Need more images ({image_count} < {min_images}), continuing to page {page + 1}")
        else:
            print(f"Reached max pages ({max_pages}), proceeding with {image_count} images")

    # Restore and move files with names not in the best group text file to the outliers folder
    restore_best_group(actor_name, 'training')

    # Step 5: Generate embeddings
    embedding_cmd = [get_venv_python(), 'compute_average_embeddings.py', actor_name]
    embeddings_success = run_subprocess_command(embedding_cmd, "Computing average embeddings")

    final_count = len(get_image_files(training_folder))
    return embeddings_success and final_count > 0, final_count


def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(description='Run actor training pipeline')
    parser.add_argument('actor_name', help='Name of the actor (e.g., "Bill Murray")')
    parser.add_argument('show_name', help='Name of the show/movie (e.g., "SNL")')
    parser.add_argument('--max-pages', type=int,
                       help='Maximum pages to download (default: from .env or 10)')
    parser.add_argument('--min-images', type=int,
                       help='Minimum training images required (default: from .env or 15)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show all output from subprocess commands')

    args = parser.parse_args()

    if args.verbose:
        os.environ['STARMAPR_LOG_VERBOSE'] = 'true'

    # Get configuration from environment or arguments
    max_pages = args.max_pages or get_env_int('MAX_DOWNLOAD_PAGES', 10)
    min_images = args.min_images or get_env_int('TRAINING_MIN_IMAGES', 15)

    print_header(f"=== TRAINING PIPELINE: {args.actor_name} ({args.show_name}) ===")
    print(f"Configuration: {min_images} min images, max {max_pages} pages")

    # Run training pipeline
    success, final_count = run_training_pipeline(
        args.actor_name, args.show_name, max_pages, min_images
    )

    if not success:
        fatal_error("Training pipeline failed!")

    print(f"✓ Training pipeline completed with {final_count} images")
    print(f"✓ Average embeddings saved to: {get_average_embedding_path(args.actor_name, 'training')}")
    sys.exit(0)


if __name__ == '__main__':
    main()
