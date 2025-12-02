#!/usr/bin/env python3
"""
StarMapr Comprehensive Actor Training Script

Orchestrates the complete training and testing pipeline for an actor by calling:
1. run_training_pipeline.py - Training phase
2. run_testing_pipeline.py - Testing phase

Usage:
    python3 run_actor_training.py "Actor Name" "Show Name"
    python3 run_actor_training.py "Actor Name" "Show Name" --retrain
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
    get_actor_folder_path, get_env_int,
    get_average_embedding_path, print_error
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
    """Main function to orchestrate the training and testing pipelines."""
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

    print_header(f"=== COMPREHENSIVE TRAINING: {args.actor_name} ({args.show_name}) ===")

    # Step 1: Run training pipeline
    print_header("\n=== STEP 1: TRAINING PIPELINE ===")
    training_cmd = [
        'venv/bin/python3', 'run_training_pipeline.py',
        args.actor_name, args.show_name
    ]
    if args.verbose:
        training_cmd.append('--verbose')

    try:
        result = subprocess.run(training_cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print_error(f"Training pipeline failed: {e}")
        fatal_error("Training pipeline failed!")

    # Step 2: Run testing pipeline
    print_header("\n=== STEP 2: TESTING PIPELINE ===")
    testing_cmd = [
        'venv/bin/python3', 'run_testing_pipeline.py',
        args.actor_name, args.show_name
    ]
    if args.verbose:
        testing_cmd.append('--verbose')

    testing_success = False
    try:
        result = subprocess.run(testing_cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        testing_success = True
    except subprocess.CalledProcessError as e:
        print_error(f"Testing pipeline failed: {e}")
        print(result.stdout if 'result' in locals() else "")

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time / 60

    # Final results
    print_header(f"\n=== FINAL RESULTS for '{args.actor_name}' ===")
    print(f"Total execution time: {elapsed_minutes:.1f} minutes ({elapsed_time:.1f} seconds)")

    if testing_success:
        print(f"🎉 SUCCESS! Both training and testing pipelines completed successfully")

        # Copy model file to models directory
        if copy_model_to_models_dir(args.actor_name):
            print("✓ Model successfully copied to models directory")
        else:
            print_error("⚠️ Warning: Failed to copy model file, but training was successful")

        sys.exit(0)
    else:
        fatal_error(f"Testing pipeline did not meet minimum requirements")


if __name__ == '__main__':
    main()
