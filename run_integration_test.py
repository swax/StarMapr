#!/usr/bin/env python3
"""
Integration Test Script for StarMapr Headshot Detection

This script tests the complete headshot detection pipeline using hardcoded mock values.
It calls run_headshot_detection.py with MOCK_VIDEO and MOCK_ACTOR as magic strings
that other scripts can detect and handle specially for testing.
"""

import subprocess
import sys
import os
import shutil
import argparse
from pathlib import Path
from utils import get_average_embedding_path, print_error

def print_header(text):
    """Print a header in yellow color."""
    yellow = '\033[93m'
    reset = '\033[0m'
    print(f"{yellow}{text}{reset}")

def verify_file_counts():
    """Verify that the mock_actor folders have the expected number of files."""
    print_header("VERIFYING FILE COUNTS")
    
    # Expected file counts based on Thomas Lennon reference
    expected_counts = {
        '02_training/mock_actor': 33,
        '02_training/mock_actor/outliers': 10,
        '02_training/mock_actor/duplicates': 4,
        '02_training/mock_actor/bad_error': 22,
        '03_testing/mock_actor': 14,
        '03_testing/mock_actor/detected_headshots': 5,
        '03_testing/mock_actor/duplicates': 2,
        '03_testing/mock_actor/bad_error': 18,
        '03_testing/mock_actor/bad_faces': 42,
        '03_testing/mock_actor/bad_unsupported': 3,
        '05_videos/mock_video': 4,
        '05_videos/mock_video/headshots': 0,
        '05_videos/mock_video/headshots/mock_actor': 5,
        '05_videos/mock_video/frames': 100
    }
    
    all_passed = True
    
    for folder_path, expected_count in expected_counts.items():
        folder = Path(folder_path)
        if not folder.exists():
            print_error(f"❌ Folder not found: {folder_path}")
            all_passed = False
            continue
        
        all_files = [f for f in folder.iterdir() if f.is_file()]
        actual_count = len(all_files)
        
        print(f"{folder_path}: {actual_count} files (expected {expected_count})")
        
        if actual_count != expected_count:
            print_error(f"❌ File count mismatch in {folder_path}")
            all_passed = False
        else:
            print(f"✅ {folder_path} correct")
    
    # Validate that the model file exists in 04_models
    pkl_path = get_average_embedding_path('mock_actor', 'models')
    if pkl_path.exists():
        print(f"✅ Model file found: {pkl_path}")
    else:
        print_error(f"❌ Model file missing: {pkl_path}")
        all_passed = False
    
    if all_passed:
        print("✅ All file counts and model file verified successfully")
    else:
        print_error("❌ Some file counts or model file did not match expectations")
    
    return all_passed

def cleanup_mock_folders():
    """Clean up existing mock folders before running test."""
    print_header("CLEANING UP MOCK FOLDERS")

    folders_to_remove = [
        '01_search_cache/mock_actor',
        '02_training/mock_actor',
        '03_testing/mock_actor',
        '05_videos/mock_video',
    ]
    
    for folder_path in folders_to_remove:
        folder = Path(folder_path)
        if folder.exists():
            print(f"Removing: {folder_path}")
            shutil.rmtree(folder)
        else:
            print(f"Not found (skipping): {folder_path}")
    
    # Remove mock actor pkl file from models folder
    pkl_path = get_average_embedding_path('mock_actor', 'models')
    if pkl_path.exists():
        print(f"Removing: {pkl_path}")
        pkl_path.unlink()
    else:
        print(f"Not found (skipping): {pkl_path}")

def seed_mock_data():
    """Copy mock data from mocks/ folder to base folder structure."""
    print_header("SEEDING MOCK DATA")
    
    mocks_folder = Path('00_mocks')
    if not mocks_folder.exists():
        print_error("❌ mocks/ folder not found")
        return False
    
    # Copy all contents from mocks/ to current directory
    for item in mocks_folder.iterdir():
        if item.is_dir():
            destination = Path(item.name)
            print(f"Copying directory: {item} -> {destination}")
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            destination = Path(item.name)
            print(f"Copying file: {item} -> {destination}")
            shutil.copy2(item, destination)
    
    print("✅ Mock data seeded successfully")
    return True

def run_integration_test(verbose=False):
    """Run the integration test with hardcoded mock parameters."""
    print_header("STARMAPR INTEGRATION TEST")
    
    # Clean up existing mock folders first
    cleanup_mock_folders()
    
    # Seed mock data
    if not seed_mock_data():
        print_error("❌ Failed to seed mock data")
        return False
    
    print("Running headshot detection with mock parameters...")
    
    # Hardcoded mock values - these are magic strings that scripts will look for
    command = [
        'python3', 
        'run_headshot_detection.py',
        'MOCK_VIDEO',
        '--show', 'MOCK_SHOW',
        '--actors', 'MOCK_ACTOR'
    ]
    
    if verbose:
        command.append('--verbose')
    
    print(f"Command: {' '.join(command)}")
    print()
    
    try:
        # Run the headshot detection script with mock parameters
        result = subprocess.run(command, check=True)
        print_header("HEADSHOT DETECTION COMPLETED")
        
        # Verify file counts
        if verify_file_counts():
            print_header("INTEGRATION TEST PASSED")
            return True
        else:
            print_error("INTEGRATION TEST FAILED - File count verification failed")
            return False
            
    except subprocess.CalledProcessError as e:
        print_error("INTEGRATION TEST FAILED")
        print(f"Exit code: {e.returncode}")
        return False
    except Exception as e:
        print_error("INTEGRATION TEST ERROR")
        print(f"Error: {e}")
        return False

def main():
    """Main function to run integration test."""
    parser = argparse.ArgumentParser(description='Run StarMapr integration test')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show all output from subprocess commands')
    
    args = parser.parse_args()
    success = run_integration_test(verbose=args.verbose)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()