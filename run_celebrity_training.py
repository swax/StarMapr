#!/usr/bin/env python3
"""
StarMapr Comprehensive Celebrity Training Script

Automated script that runs the complete training and testing pipeline for a celebrity.
Iteratively downloads and processes images until minimum thresholds are met.

Training: Downloads up to 5 pages until 15+ quality training images
Testing: Downloads up to 5 pages until 4+ headshots are detected

Usage:
    python3 train_celebrity_comprehensive.py "Celebrity Name" "Show Name"
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
    get_celebrity_folder_path, get_celebrity_folder_name, get_image_files, get_env_int,
    get_average_embedding_path, print_error, print_run_summary, ensure_folder_exists
)

# Load environment variables
load_dotenv()


def count_detected_headshots(celebrity_name):
    """
    Count the number of detected headshots for a celebrity.
    
    Args:
        celebrity_name (str): Name of the celebrity
        
    Returns:
        int: Number of detected headshots
    """
    testing_folder = get_celebrity_folder_path(celebrity_name, 'testing')
    headshots_folder = Path(testing_folder) / 'detected_headshots'
    
    if not headshots_folder.exists():
        return 0
    
    headshot_files = get_image_files(headshots_folder)
    return len(headshot_files)


def fatal_error(message):
    print_error(f"❌ {message}")
    sys.exit(1)


def copy_model_to_models_dir(celebrity_name):
    """
    Copy the average embedding file from training directory to models directory.
    
    Args:
        celebrity_name (str): Name of the celebrity
        
    Returns:
        bool: True if successful, False if failed
    """
    try:
        # Get paths using utility functions
        source_path = get_average_embedding_path(celebrity_name, 'training')
        dest_path = get_average_embedding_path(celebrity_name, 'models')
        
        # Create models directory if it doesn't exist
        dest_path.parent.mkdir(exist_ok=True)
        
        # Copy file to models directory
        shutil.copy2(source_path, dest_path)
        
        print_run_summary(f"✓ Copied model to: {dest_path}")
        return True
        
    except Exception as e:
        print_error(f"Failed to copy model file: {e}")
        return False


def delete_existing_folders(celebrity_name):
    """Start fresh in case the last run failed and training is in an incomplete state."""
    training_folder = get_celebrity_folder_path(celebrity_name, 'training')
    testing_folder = get_celebrity_folder_path(celebrity_name, 'testing')
    
    folders_exist = os.path.exists(training_folder) or os.path.exists(testing_folder)
    if not folders_exist:
        print(f"No existing folders found for '{celebrity_name}', proceeding with training.")
        return
    
    for folder in [training_folder, testing_folder]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    
    print_run_summary(f"✓ Deleted existing folders for '{celebrity_name}'")


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
        result = subprocess.run(command_list, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed: {description}")
        if e.stderr:
            print_error(e.stderr.strip())
        return False


def run_training_pipeline(celebrity_name, show_name, max_pages, min_images):
    """
    Run the training pipeline until minimum images achieved or max pages reached.
    
    Args:
        celebrity_name (str): Name of the celebrity
        show_name (str): Name of the show
        max_pages (int): Maximum pages to download
        min_images (int): Minimum training images required
        
    Returns:
        tuple: (success: bool, final_image_count: int)
    """
    print_run_summary(f"\n=== TRAINING PIPELINE for '{celebrity_name}' ===")
    
    training_folder = get_celebrity_folder_path(celebrity_name, 'training')
    ensure_folder_exists(training_folder)
    
    for page in range(1, max_pages + 1):
        print_run_summary(f"\n--- Training Page {page} ---")
        
        # Step 1: Download training images
        download_cmd = [
            'python3', 'download_celebrity_images.py', celebrity_name,
            '--training', '--show', show_name, '--page', str(page)
        ]
        if not run_subprocess_command(download_cmd, f"Downloading training images (page {page})"):
            fatal_error(f"Failed to download training images for page {page}")
        
        # Step 2: Remove duplicates
        dedup_cmd = ['python3', 'remove_dupe_training_images.py', '--training', celebrity_name]
        if not run_subprocess_command(dedup_cmd, "Removing duplicate images"):
            fatal_error("Failed to remove duplicate images")
        
        # Step 3: Remove bad images (not exactly 1 face)
        bad_cmd = ['python3', 'remove_bad_training_images.py', '--training', celebrity_name]
        if not run_subprocess_command(bad_cmd, "Removing bad training images"):
            fatal_error("Failed to remove bad training images")
        
        # Step 4: Remove face outliers
        outlier_cmd = ['python3', 'remove_face_outliers.py', '--training', celebrity_name]
        if not run_subprocess_command(outlier_cmd, "Removing face outliers"):
            fatal_error("Failed to remove face outliers")
        
        # Count remaining images
        current_images = get_image_files(training_folder)
        image_count = len(current_images)
        
        print_run_summary(f"Training images after page {page}: {image_count}")
        
        if image_count >= min_images:
            print_run_summary(f"✓ Achieved minimum training images ({image_count} >= {min_images})")
            break
        elif page < max_pages:
            print(f"Need more images ({image_count} < {min_images}), continuing to page {page + 1}")
        else:
            print(f"Reached max pages ({max_pages}), proceeding with {image_count} images")
    
    # Step 5: Generate embeddings
    embedding_cmd = ['python3', 'compute_average_embeddings.py', celebrity_name]
    embeddings_success = run_subprocess_command(embedding_cmd, "Computing average embeddings")
    
    final_count = len(get_image_files(training_folder))
    return embeddings_success and final_count > 0, final_count


def run_testing_pipeline(celebrity_name, show_name, max_pages, min_headshots):
    """
    Run the testing pipeline until minimum headshots detected or max pages reached.
    
    Args:
        celebrity_name (str): Name of the celebrity
        show_name (str): Name of the show
        max_pages (int): Maximum pages to download
        min_headshots (int): Minimum headshots required for success
        
    Returns:
        tuple: (success: bool, final_headshot_count: int)
    """
    print_run_summary(f"\n=== TESTING PIPELINE for '{celebrity_name}' ===")
    
    testing_folder = get_celebrity_folder_path(celebrity_name, 'testing')
    ensure_folder_exists(testing_folder)
    
    for page in range(1, max_pages + 1):
        print_run_summary(f"\n--- Testing Page {page} ---")
        
        # Step 1: Download testing images
        download_cmd = [
            'python3', 'download_celebrity_images.py', celebrity_name,
            '--testing', '--show', show_name, '--page', str(page)
        ]
        if not run_subprocess_command(download_cmd, f"Downloading testing images (page {page})"):
            fatal_error(f"Failed to download testing images for page {page}")
        
        # Step 2: Remove duplicates
        dedup_cmd = ['python3', 'remove_dupe_training_images.py', '--testing', celebrity_name]
        if not run_subprocess_command(dedup_cmd, "Removing duplicate images"):
            fatal_error("Failed to remove duplicate images")
        
        # Step 3: Remove bad images (not 4-10 faces)
        bad_cmd = ['python3', 'remove_bad_training_images.py', '--testing', celebrity_name]
        if not run_subprocess_command(bad_cmd, "Removing bad testing images"):
            fatal_error("Failed to remove bad testing images")
        
        # Step 4: Run face detection
        detect_cmd = ['python3', 'eval_star_detection.py', celebrity_name]
        if not run_subprocess_command(detect_cmd, "Running face detection"):
            fatal_error("Failed to run face detection")
        
        # Count detected headshots
        headshot_count = count_detected_headshots(celebrity_name)
        print_run_summary(f"Detected headshots after page {page}: {headshot_count}")
        
        if headshot_count >= min_headshots:
            print_run_summary(f"✓ Achieved minimum headshots ({headshot_count} >= {min_headshots})")
            return True, headshot_count
        elif page < max_pages:
            print(f"Need more headshots ({headshot_count} < {min_headshots}), continuing to page {page + 1}")
        else:
            print(f"Reached max pages ({max_pages}), final headshots: {headshot_count}")
    
    final_headshots = count_detected_headshots(celebrity_name)
    return final_headshots >= min_headshots, final_headshots


def check_existing_model(celebrity_name):
    """
    Check if a model already exists for the celebrity.
    
    Args:
        celebrity_name (str): Name of the celebrity
        
    Returns:
        bool: True if model exists, False otherwise
    """
    model_path = get_average_embedding_path(celebrity_name, 'models')
    return model_path.exists()


def main():
    """Main function to run the comprehensive training pipeline."""
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Run comprehensive celebrity training pipeline')
    parser.add_argument('celebrity_name', help='Name of the celebrity (e.g., "Bill Murray")')
    parser.add_argument('show_name', help='Name of the show/movie (e.g., "SNL")')
    parser.add_argument('--retrain', action='store_true', help='Delete existing celebrity folders before starting')
    
    args = parser.parse_args()
    
    # Check if model already exists (unless using --retrain flag)
    if not args.retrain and check_existing_model(args.celebrity_name):
        model_path = get_average_embedding_path(args.celebrity_name, 'models')
        print_run_summary(f"✓ Model already exists: {model_path}")
        print_run_summary(f"Skipping training for '{args.celebrity_name}' (use --retrain to retrain)")
        sys.exit(0)
    
    # Clean any previous failed runs
    delete_existing_folders(args.celebrity_name)
    
    # Get configuration from environment
    min_training_images = get_env_int('TRAINING_MIN_IMAGES', 15)
    min_testing_headshots = get_env_int('TESTING_MIN_HEADSHOTS', 4)
    max_pages = get_env_int('MAX_DOWNLOAD_PAGES', 5)
    
    print_run_summary(f"=== COMPREHENSIVE TRAINING: {args.celebrity_name} ({args.show_name}) ===")
    print(f"Configuration: {min_training_images} training images, {min_testing_headshots} headshots, max {max_pages} pages")
    
    # Run training pipeline
    training_success, training_count = run_training_pipeline(
        args.celebrity_name, args.show_name, max_pages, min_training_images
    )
    
    if not training_success:
        fatal_error("Training pipeline failed!")
    
    print_run_summary(f"✓ Training pipeline completed with {training_count} images")
    
    # Run testing pipeline
    testing_success, headshot_count = run_testing_pipeline(
        args.celebrity_name, args.show_name, max_pages, min_testing_headshots
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time / 60
    
    # Final results
    print_run_summary(f"\n=== FINAL RESULTS for '{args.celebrity_name}' ===")
    print(f"Training images: {training_count}")
    print(f"Detected headshots: {headshot_count}")
    print(f"Total execution time: {elapsed_minutes:.1f} minutes ({elapsed_time:.1f} seconds)")
    
    if testing_success:
        print_run_summary(f"🎉 SUCCESS! Found {headshot_count} headshots (>= {min_testing_headshots} required)")
        
        # Copy model file to models directory
        if copy_model_to_models_dir(args.celebrity_name):
            print_run_summary("✓ Model successfully copied to models directory")
        else:
            print_error("⚠️ Warning: Failed to copy model file, but training was successful")
        
        sys.exit(0)
    else:
        fatal_error(f"FAILED! Only found {headshot_count} headshots (< {min_testing_headshots} required)")


if __name__ == '__main__':
    main()