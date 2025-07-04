#!/usr/bin/env python3
"""
Integration Test Script for StarMapr Headshot Detection

This script tests the complete headshot detection pipeline using hardcoded mock values.
It calls run_headshot_detection.py with MOCK_VIDEO and MOCK_CELEBRITY as magic strings
that other scripts can detect and handle specially for testing.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path
from utils import get_average_embedding_path

def verify_file_counts():
    """Verify that the mock_celebrity folders have the expected number of files."""
    print("=== VERIFYING FILE COUNTS ===")
    
    # Expected file counts based on Thomas Lennon reference
    expected_counts = {
        'training/mock_celebrity': 33,
        'training/mock_celebrity/outliers': 10,
        'training/mock_celebrity/duplicates': 4,
        'training/mock_celebrity/bad_error': 22,
        'testing/mock_celebrity': 14,
        'testing/mock_celebrity/detected_headshots': 5,
        'testing/mock_celebrity/duplicates': 2,
        'testing/mock_celebrity/bad_error': 18,
        'testing/mock_celebrity/bad_faces': 42,
        'testing/mock_celebrity/bad_unsupported': 3,
        'videos/mock_video': 4,
        'videos/mock_video/headshots': 0,
        'videos/mock_video/headshots/mock_celebrity': 5,
        'videos/mock_video/frames': 100
    }
    
    all_passed = True
    
    for folder_path, expected_count in expected_counts.items():
        folder = Path(folder_path)
        if not folder.exists():
            print(f"❌ Folder not found: {folder_path}")
            all_passed = False
            continue
        
        all_files = [f for f in folder.iterdir() if f.is_file()]
        actual_count = len(all_files)
        
        print(f"{folder_path}: {actual_count} files (expected {expected_count})")
        
        if actual_count != expected_count:
            print(f"❌ File count mismatch in {folder_path}")
            all_passed = False
        else:
            print(f"✅ {folder_path} correct")
    
    if all_passed:
        print("✅ All file counts verified successfully")
    else:
        print("❌ Some file counts did not match expectations")
    
    return all_passed

def cleanup_mock_folders():
    """Clean up existing mock folders before running test."""
    print("=== CLEANING UP MOCK FOLDERS ===")

    folders_to_remove = [
        'search_cache/mock_celebrity',
        'training/mock_celebrity',
        'testing/mock_celebrity',
        'videos/mock_video',
    ]
    
    for folder_path in folders_to_remove:
        folder = Path(folder_path)
        if folder.exists():
            print(f"Removing: {folder_path}")
            shutil.rmtree(folder)
        else:
            print(f"Not found (skipping): {folder_path}")
    
    # Remove mock celebrity pkl file from models folder
    pkl_path = get_average_embedding_path('mock_celebrity', 'models')
    if pkl_path.exists():
        print(f"Removing: {pkl_path}")
        pkl_path.unlink()
    else:
        print(f"Not found (skipping): {pkl_path}")

def seed_mock_data():
    """Copy mock data from mocks/ folder to base folder structure."""
    print("=== SEEDING MOCK DATA ===")
    
    mocks_folder = Path('mocks')
    if not mocks_folder.exists():
        print("❌ mocks/ folder not found")
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

def run_integration_test():
    """Run the integration test with hardcoded mock parameters."""
    print("=== STARMAPR INTEGRATION TEST ===")
    
    # Clean up existing mock folders first
    cleanup_mock_folders()
    
    # Seed mock data
    if not seed_mock_data():
        print("❌ Failed to seed mock data")
        return False
    
    print("Running headshot detection with mock parameters...")
    
    # Hardcoded mock values - these are magic strings that scripts will look for
    command = [
        'python3', 
        'run_headshot_detection.py',
        'MOCK_VIDEO',
        '--show', 'MOCK_SHOW',
        '--celebrities', 'MOCK_CELEBRITY'
    ]
    
    print(f"Command: {' '.join(command)}")
    print()
    
    try:
        # Run the headshot detection script with mock parameters
        result = subprocess.run(command, check=True)
        print("\n=== HEADSHOT DETECTION COMPLETED ===")
        
        # Verify file counts
        if verify_file_counts():
            print("\n=== INTEGRATION TEST PASSED ===")
            return True
        else:
            print("\n=== INTEGRATION TEST FAILED - File count verification failed ===")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n=== INTEGRATION TEST FAILED ===")
        print(f"Exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n=== INTEGRATION TEST ERROR ===")
        print(f"Error: {e}")
        return False

def main():
    """Main function to run integration test."""
    success = run_integration_test()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()