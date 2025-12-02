#!/usr/bin/env python3
"""
StarMapr Comprehensive Actor Training Script

Automated script that runs the complete training and testing pipeline for a actor.
Iteratively downloads and processes images until minimum thresholds are met.

Training: Downloads up to 5 pages until 15+ quality training images
Testing: Downloads up to 5 pages until 4+ headshots are detected

Usage:
    python3 train_actor_comprehensive.py "Actor Name" "Show Name"
"""

import os
import sys
import subprocess
import argparse
import shutil
import time
from pathlib import Path
from dotenv import load_dotenv
from utils import (
    get_actor_folder_path, get_image_files, get_env_int,
    get_average_embedding_path, print_error, ensure_folder_exists
)

# Load environment variables
load_dotenv()

def print_header(text):
    """Print a header in blue color."""
    blue = '\033[94m'
    reset = '\033[0m'
    print(f"{blue}{text}{reset}")

def count_detected_headshots(actor_name):
    """
    Count the number of detected headshots for a actor.
    
    Args:
        actor_name (str): Name of the actor
        
    Returns:
        int: Number of detected headshots
    """
    testing_folder = get_actor_folder_path(actor_name, 'testing')
    headshots_folder = Path(testing_folder) / 'detected_headshots'
    
    if not headshots_folder.exists():
        return 0
    
    headshot_files = get_image_files(headshots_folder)
    return len(headshot_files)


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

    #print(f"✓ Moved {files_moved} files (images + pkl) back from outliers folder")
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


def copy_model_to_models_dir(actor_name):
    """
    Copy the average embedding file from training directory to models directory.

    Args:
        actor_name (str): Name of the actor

    Returns:
        bool: True if successful, False if failed
    """
    try:
        # Get paths using utility functions
        source_path = get_average_embedding_path(actor_name, 'training')
        dest_path = get_average_embedding_path(actor_name, 'models')

        # Create models directory if it doesn't exist
        dest_path.parent.mkdir(exist_ok=True)

        # Copy file to models directory
        shutil.copy2(source_path, dest_path)

        print(f"✓ Copied model to: {dest_path}")
        return True

    except Exception as e:
        print_error(f"Failed to copy model file: {e}")
        return False


def delete_existing_folders(actor_name):
    """Start fresh in case the last run failed and training is in an incomplete state."""
    training_folder = get_actor_folder_path(actor_name, 'training')
    testing_folder = get_actor_folder_path(actor_name, 'testing')
    
    folders_exist = os.path.exists(training_folder) or os.path.exists(testing_folder)
    if not folders_exist:
        print(f"No existing folders found for '{actor_name}', proceeding with training.")
        return
    
    for folder in [training_folder, testing_folder]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    
    print(f"✓ Deleted existing folders for '{actor_name}'")


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
        result = subprocess.run(command_list, check=True, capture_output=True, text=True)
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
            'venv/bin/python3', 'download_actor_images.py', actor_name,
            '--training', '--show', show_name, '--page', str(page)
        ]
        if not run_subprocess_command(download_cmd, f"Downloading training images (page {page})"):
            fatal_error(f"Failed to download training images for page {page}")
        
        # Step 2: Remove duplicates
        dedup_cmd = ['venv/bin/python3', 'remove_dupe_training_images.py', '--training', actor_name]
        if not run_subprocess_command(dedup_cmd, "Removing duplicate images"):
            fatal_error("Failed to remove duplicate images")
        
        # Step 3: Remove bad face counts (not exactly 1 face)
        bad_cmd = ['venv/bin/python3', 'remove_bad_training_images.py', '--training', actor_name]
        if not run_subprocess_command(bad_cmd, "Removing bad training images"):
            fatal_error("Failed to remove bad training images")

        current_images = get_image_files(training_folder)
        image_count = len(current_images)
        if image_count == 0:
            print_header("No images left after cleaning, continuing to next page...")
            continue

        # Step 4a: Try similarity-based outlier detection first (works better with fewer images)
        restore_outliers_to_training(actor_name, 'training')

        outlier_cmd = ['venv/bin/python3', 'remove_face_outliers.py', '--training', actor_name]
        if not run_subprocess_command(outlier_cmd, "Removing face outliers (similarity based)"):
            fatal_error("Failed to remove face outliers")

        threshold_met, image_count, best_image_count = check_image_threshold(
            training_folder, min_images, best_image_count
        )
        if threshold_met:
            break

        # Step 4b: Try DBSCAN clustering, works better with lots of images where there might be an island of good ones that all match
        restore_outliers_to_training(actor_name, 'training')

        cluster_cmd = ['venv/bin/python3', 'cluster_and_keep_largest.py', '--training', actor_name]
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
    embedding_cmd = ['venv/bin/python3', 'compute_average_embeddings.py', actor_name]
    embeddings_success = run_subprocess_command(embedding_cmd, "Computing average embeddings")
    
    final_count = len(get_image_files(training_folder))
    return embeddings_success and final_count > 0, final_count


def run_testing_pipeline(actor_name, show_name, max_pages, min_headshots):
    """
    Run the testing pipeline until minimum headshots detected or max pages reached.
    
    Args:
        actor_name (str): Name of the actor
        show_name (str): Name of the show
        max_pages (int): Maximum pages to download
        min_headshots (int): Minimum headshots required for success
        
    Returns:
        tuple: (success: bool, final_headshot_count: int)
    """
    print_header(f"\n=== TESTING PIPELINE for '{actor_name}' ===")
    
    testing_folder = get_actor_folder_path(actor_name, 'testing')
    ensure_folder_exists(testing_folder)
    
    for page in range(1, max_pages + 1):
        print_header(f"\n--- Testing Page {page} ---")
        
        # Step 1: Download testing images
        download_cmd = [
            'venv/bin/python3', 'download_actor_images.py', actor_name,
            '--testing', '--show', show_name, '--page', str(page)
        ]
        if not run_subprocess_command(download_cmd, f"Downloading testing images (page {page})"):
            fatal_error(f"Failed to download testing images for page {page}")
        
        # Step 2: Remove duplicates
        dedup_cmd = ['venv/bin/python3', 'remove_dupe_training_images.py', '--testing', actor_name]
        if not run_subprocess_command(dedup_cmd, "Removing duplicate images"):
            fatal_error("Failed to remove duplicate images")
        
        # Step 3: Remove bad face counts (not 4-10 faces)
        bad_cmd = ['venv/bin/python3', 'remove_bad_training_images.py', '--testing', actor_name]
        if not run_subprocess_command(bad_cmd, "Removing bad testing images"):
            fatal_error("Failed to remove bad testing images")
        
        # Step 4: Run face detection
        detect_cmd = ['venv/bin/python3', 'eval_star_detection.py', actor_name]
        if not run_subprocess_command(detect_cmd, "Running face detection"):
            fatal_error("Failed to run face detection")
        
        # Count detected headshots
        headshot_count = count_detected_headshots(actor_name)
        print_header(f"Detected headshots after page {page}: {headshot_count}")
        
        if headshot_count >= min_headshots:
            print(f"✓ Achieved minimum headshots ({headshot_count} >= {min_headshots})")
            return True, headshot_count
        elif page < max_pages:
            print(f"Need more headshots ({headshot_count} < {min_headshots}), continuing to page {page + 1}")
        else:
            print(f"Reached max pages ({max_pages}), final headshots: {headshot_count}")
    
    final_headshots = count_detected_headshots(actor_name)
    return final_headshots >= min_headshots, final_headshots


def check_existing_model(actor_name):
    """
    Check if a model already exists for the actor.
    
    Args:
        actor_name (str): Name of the actor
        
    Returns:
        bool: True if model exists, False otherwise
    """
    model_path = get_average_embedding_path(actor_name, 'models')
    return model_path.exists()


def main():
    """Main function to run the comprehensive training pipeline."""
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Run comprehensive actor training pipeline')
    parser.add_argument('actor_name', help='Name of the actor (e.g., "Bill Murray")')
    parser.add_argument('show_name', help='Name of the show/movie (e.g., "SNL")')
    parser.add_argument('--retrain', action='store_true', help='Delete existing actor folders before starting')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show all output from subprocess commands')
    
    args = parser.parse_args()

    if args.verbose:
        os.environ['STARMAPR_LOG_VERBOSE'] = 'true'

    # Check if model already exists (unless using --retrain flag)
    if not args.retrain and check_existing_model(args.actor_name):
        model_path = get_average_embedding_path(args.actor_name, 'models')
        print(f"✓ Model already exists: {model_path}")
        print(f"Skipping training for '{args.actor_name}' (use --retrain to retrain)")
        sys.exit(0)
    
    # Clean any previous failed runs
    delete_existing_folders(args.actor_name)
    
    # Get configuration from environment
    min_training_images = get_env_int('TRAINING_MIN_IMAGES', 15)
    min_testing_headshots = get_env_int('TESTING_MIN_HEADSHOTS', 4)
    max_pages = get_env_int('MAX_DOWNLOAD_PAGES', 10)
    
    print_header(f"=== COMPREHENSIVE TRAINING: {args.actor_name} ({args.show_name}) ===")
    print(f"Configuration: {min_training_images} training images, {min_testing_headshots} headshots, max {max_pages} pages")
    
    # Run training pipeline
    training_success, training_count = run_training_pipeline(
        args.actor_name, args.show_name, max_pages, min_training_images
    )
    
    if not training_success:
        fatal_error("Training pipeline failed!")
    
    print(f"✓ Training pipeline completed with {training_count} images")
    
    # Run testing pipeline
    testing_success, headshot_count = run_testing_pipeline(
        args.actor_name, args.show_name, max_pages, min_testing_headshots
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time / 60
    
    # Final results
    print_header(f"\n=== FINAL RESULTS for '{args.actor_name}' ===")
    print(f"Training images: {training_count}")
    print(f"Detected headshots: {headshot_count}")
    print(f"Total execution time: {elapsed_minutes:.1f} minutes ({elapsed_time:.1f} seconds)")
    
    if testing_success:
        print(f"🎉 SUCCESS! Found {headshot_count} headshots (>= {min_testing_headshots} required)")
        
        # Copy model file to models directory
        if copy_model_to_models_dir(args.actor_name):
            print("✓ Model successfully copied to models directory")
        else:
            print_error("⚠️ Warning: Failed to copy model file, but training was successful")
        
        sys.exit(0)
    else:
        fatal_error(f"FAILED! Only found {headshot_count} headshots (< {min_testing_headshots} required)")


if __name__ == '__main__':
    main()
