#!/usr/bin/env python3
"""
StarMapr Comprehensive Actor Training Script

Orchestrates the complete training and testing pipeline for an actor by calling:
1. 03_run_training_pipeline.py - Training phase
2. 04_run_testing_pipeline.py - Testing phase

Usage:
    python3 02_run_actor_training.py "Actor Name" "Show Name"
    python3 02_run_actor_training.py "Actor Name" "Show Name" --retrain
"""

import os
import sys
import subprocess
import argparse
import shutil
import time
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from utils import (
    get_actor_folder_path, get_env_int,
    get_average_embedding_path, print_error, get_venv_python
)
from validation import metadata_path, validate_model_metadata, write_json
from progress import run_logged, emit_progress

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

        report = validate_model_metadata(source_path)
        if report is None:
            raise ValueError('Missing, stale, or rejected model quality report')
        # Stage first. A failed retrain leaves the old model untouched. Hash checks
        # make the short model/report replacement interval fail closed for readers.
        with tempfile.TemporaryDirectory(dir=dest_path.parent) as staging:
            staged = Path(staging) / dest_path.name
            shutil.copy2(source_path, staged)
            staged_report = metadata_path(staged)
            write_json(staged_report, report)
            previous = Path(staging) / 'previous.pkl'
            if dest_path.exists():
                shutil.copy2(dest_path, previous)
            os.replace(staged, dest_path)
            try:
                os.replace(staged_report, metadata_path(dest_path))
            except OSError:
                if previous.exists():
                    os.replace(previous, dest_path)
                else:
                    dest_path.unlink(missing_ok=True)
                raise

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

    for folder in map(Path, [training_folder, testing_folder]):
        if folder.exists():
            archive = folder.parent / '.history' / (folder.name + '-' + str(time.time_ns()))
            archive.parent.mkdir(exist_ok=True)
            folder.rename(archive)
            folder.mkdir()
            # Retrying training must not silently reset its durable AWS cap/cache.
            cache = archive / 'celebrity-cache.sqlite'
            if cache.exists():
                shutil.copy2(cache, folder / cache.name)

    print(f"✓ Archived existing training/testing folders for '{actor_name}'")


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


def reusable_model(actor_name):
    report = validate_model_metadata(get_average_embedding_path(actor_name, 'models'))
    if not report:
        return False
    if os.getenv('CELEBRITY_VERIFIER', 'off') == 'aws_required':
        seed = report.get('seed_identity', {})
        return (report.get('identity_validation') == 'aws_seed_verified'
                and seed.get('expected_name') == actor_name
                and seed.get('confidence', 0) >= float(os.getenv('AWS_CELEBRITY_MIN_CONFIDENCE', '99')))
    return True


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
    os.environ['STARMAPR_ACTOR'] = args.actor_name
    if not args.retrain and reusable_model(args.actor_name):
        model_path = get_average_embedding_path(args.actor_name, 'models')
        print(f"✓ Model already exists: {model_path}")
        print(f"Skipping training for '{args.actor_name}' (use --retrain to retrain)")
        sys.exit(0)

    if check_existing_model(args.actor_name):
        emit_progress('model_migration', 'started')
        print('Retraining legacy/stale model automatically; previous model is retained until promotion')

    # Clean any previous failed runs
    delete_existing_folders(args.actor_name)

    print_header(f"=== COMPREHENSIVE TRAINING: {args.actor_name} ({args.show_name}) ===")

    # Step 1: Run training pipeline
    print_header("\n=== STEP 1: TRAINING PIPELINE ===")
    training_cmd = [
        get_venv_python(), '03_run_training_pipeline.py',
        args.actor_name, args.show_name
    ]
    if args.verbose:
        training_cmd.append('--verbose')

    try:
        result = run_logged(training_cmd, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print_error(f"Training pipeline failed: {e}")
        print_error((e.stdout or '') + (e.stderr or ''))
        if e.returncode == 2:
            print('Training abstained; retaining any prior model and continuing with optional headshots')
            sys.exit(0)
        fatal_error("Training pipeline failed!")

    # Step 2: Run testing pipeline
    print_header("\n=== STEP 2: TESTING PIPELINE ===")
    testing_cmd = [
        get_venv_python(), '04_run_testing_pipeline.py',
        args.actor_name, args.show_name
    ]
    if args.verbose:
        testing_cmd.append('--verbose')

    testing_success = False
    try:
        result = run_logged(testing_cmd, check=True)
        print(result.stdout)
        testing_success = True
    except subprocess.CalledProcessError as e:
        print_error(f"Testing pipeline failed: {e}")
        print_error((e.stdout or '') + (e.stderr or ''))

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
            fatal_error('Model promotion failed')

        sys.exit(0)
    else:
        fatal_error(f"Testing pipeline did not meet minimum requirements")


if __name__ == '__main__':
    main()
