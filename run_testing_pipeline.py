#!/usr/bin/env python3
"""
StarMapr Testing Pipeline Script

Runs the complete testing pipeline for an actor:
- Downloads testing images iteratively until minimum headshots detected
- Removes duplicates and bad images
- Runs face detection to validate model accuracy

Usage:
    python3 run_testing_pipeline.py "Actor Name" "Show Name"
    python3 run_testing_pipeline.py "Actor Name" "Show Name" --max-pages 5 --min-headshots 6
"""

import os
import sys
import subprocess
import argparse
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


def fatal_error(message):
    print_error(f"❌ {message}")
    sys.exit(1)


def count_detected_headshots(actor_name):
    """
    Count the number of detected headshots for an actor.

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


def check_embeddings_exist(actor_name):
    """
    Check if average embeddings exist for the actor.

    Args:
        actor_name (str): Name of the actor

    Returns:
        bool: True if embeddings exist, False otherwise
    """
    embeddings_path = get_average_embedding_path(actor_name, 'training')
    return embeddings_path.exists()


def main():
    """Main function to run the testing pipeline."""
    parser = argparse.ArgumentParser(description='Run actor testing pipeline')
    parser.add_argument('actor_name', help='Name of the actor (e.g., "Bill Murray")')
    parser.add_argument('show_name', help='Name of the show/movie (e.g., "SNL")')
    parser.add_argument('--max-pages', type=int,
                       help='Maximum pages to download (default: from .env or 10)')
    parser.add_argument('--min-headshots', type=int,
                       help='Minimum headshots required (default: from .env or 4)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show all output from subprocess commands')

    args = parser.parse_args()

    if args.verbose:
        os.environ['STARMAPR_LOG_VERBOSE'] = 'true'

    # Check if embeddings exist before running testing
    if not check_embeddings_exist(args.actor_name):
        fatal_error(f"No average embeddings found for '{args.actor_name}'. Run training pipeline first.")

    # Get configuration from environment or arguments
    max_pages = args.max_pages or get_env_int('MAX_DOWNLOAD_PAGES', 10)
    min_headshots = args.min_headshots or get_env_int('TESTING_MIN_HEADSHOTS', 4)

    print_header(f"=== TESTING PIPELINE: {args.actor_name} ({args.show_name}) ===")
    print(f"Configuration: {min_headshots} min headshots, max {max_pages} pages")

    # Run testing pipeline
    success, final_count = run_testing_pipeline(
        args.actor_name, args.show_name, max_pages, min_headshots
    )

    print_header(f"\n=== FINAL RESULTS for '{args.actor_name}' ===")
    print(f"Detected headshots: {final_count}")

    if success:
        print(f"🎉 SUCCESS! Found {final_count} headshots (>= {min_headshots} required)")
        sys.exit(0)
    else:
        fatal_error(f"FAILED! Only found {final_count} headshots (< {min_headshots} required)")


if __name__ == '__main__':
    main()
