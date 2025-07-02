#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np
from collections import defaultdict
from utils import add_training_testing_args, get_mode_and_path_from_args, print_dry_run_header, print_dry_run_summary, get_supported_image_extensions, print_error, print_summary, move_file_with_pkl
from utils_deepface import get_face_embeddings

def count_faces_in_image(image_path):
    """
    Count the number of faces detected in an image using DeepFace.
    
    Args:
        image_path (Path): Path to the image file
        
    Returns:
        int: Number of faces detected, -1 if error occurred
    """
    face_analysis = get_face_embeddings(image_path, enforce_detection=False)
    return len(face_analysis) if face_analysis else -1

def remove_bad_images(celebrity_folder_path, mode='training', dry_run=False):
    """
    Move images from celebrity folder that don't meet face count requirements to a 'bad' subfolder.
    
    Args:
        celebrity_folder_path (str): Path to celebrity folder containing training or testing images
        mode (str): 'training' expects exactly 1 face, 'testing' expects 4-10 faces
        dry_run (bool): If True, only report what would be moved without actually moving files
    """
    folder_path = Path(celebrity_folder_path)
    
    if not folder_path.exists():
        print_error(f"Folder not found: {folder_path}")
        return
    
    # Create bad folder if it doesn't exist
    bad_folder = folder_path / "bad"
    
    # Get all files (excluding those already in bad folder and subdirectories)
    image_extensions = get_supported_image_extensions()
    all_files = [f for f in folder_path.iterdir() 
                 if f.is_file() and not f.name.startswith('.')]
    
    # Separate supported and unsupported files
    image_files = [f for f in all_files if f.suffix.lower() in image_extensions]
    unsupported_files = [f for f in all_files if f.suffix.lower() not in image_extensions]
    
    if not all_files:
        print_error(f"No files found in {folder_path}")
        return
    
    print(f"Found {len(all_files)} files in {folder_path}")
    print(f"  - {len(image_files)} supported image files")
    print(f"  - {len(unsupported_files)} unsupported format files")
    if dry_run:
        print_dry_run_header("No files will be moved")
    print()
    
    images_to_move = []
    images_with_good_faces = []
    images_with_errors = []
    unsupported_to_move = unsupported_files.copy()
    
    # Define face count requirements based on mode
    if mode == 'training':
        required_faces = 1
        face_description = "exactly 1 face"
    else:  # testing mode
        min_faces, max_faces = 3, 10
        face_description = f"{min_faces}-{max_faces} faces"
    
    for img_file in image_files:
        print(f"Analyzing: {img_file.name}")
        face_count = count_faces_in_image(img_file)
        
        if face_count == -1:
            print(f"  → ERROR: Could not process image")
            images_with_errors.append(img_file)
        elif mode == 'training':
            if face_count == required_faces:
                print(f"  → KEEP: Exactly 1 face detected")
                images_with_good_faces.append(img_file)
            else:
                if face_count == 0:
                    print(f"  → MOVE TO BAD: No faces detected")
                else:
                    print(f"  → MOVE TO BAD: {face_count} faces detected (expected exactly 1)")
                images_to_move.append(img_file)
        else:  # testing mode
            if min_faces <= face_count <= max_faces:
                print(f"  → KEEP: {face_count} faces detected (within {min_faces}-{max_faces} range)")
                images_with_good_faces.append(img_file)
            else:
                if face_count == 0:
                    print(f"  → MOVE TO BAD: No faces detected")
                elif face_count < min_faces:
                    print(f"  → MOVE TO BAD: {face_count} faces detected (need at least {min_faces})")
                else:
                    print(f"  → MOVE TO BAD: {face_count} faces detected (maximum {max_faces} allowed)")
                images_to_move.append(img_file)
    
    # Handle unsupported files
    if unsupported_to_move:
        print(f"\nUnsupported format files (will be moved to bad folder):")
        for file in unsupported_to_move:
            print(f"  → MOVE TO BAD: {file.name} (unsupported format: {file.suffix or 'no extension'})")
    
    # Summary
    print(f"\nSummary:")
    print(f"Total files found: {len(all_files)}")
    print(f"Supported images analyzed: {len(image_files)}")
    print(f"Images with {face_description} (keeping): {len(images_with_good_faces)}")
    print(f"Images to move to bad folder (wrong face count): {len(images_to_move)}")
    print(f"Unsupported files to move to bad folder: {len(unsupported_to_move)}")
    print(f"Images with processing errors: {len(images_with_errors)}")
    
    all_files_to_move = images_to_move + unsupported_to_move
    
    if images_to_move:
        print(f"\nImages to be moved to bad folder (wrong face count):")
        for img in images_to_move:
            print(f"  - {img.name}")
    
    if unsupported_to_move:
        print(f"\nUnsupported files to be moved to bad folder:")
        for file in unsupported_to_move:
            print(f"  - {file.name}")
    
    if images_with_errors:
        print(f"\nImages with processing errors (not moved):")
        for img in images_with_errors:
            print(f"  - {img.name}")
    
    # Move bad images and unsupported files if not in dry run mode
    if all_files_to_move and not dry_run:
        print(f"\nMoving {len(all_files_to_move)} files to {bad_folder}...")
        
        total_moved_count = 0
        total_attempted_count = 0
        
        for file_to_move in all_files_to_move:
            moved_count, attempted_count = move_file_with_pkl(file_to_move, bad_folder, dry_run)
            total_moved_count += moved_count
            total_attempted_count += attempted_count
        
        print_summary(f"Successfully moved {total_moved_count}/{total_attempted_count} files to bad folder")
    
    elif all_files_to_move and dry_run:
        total_files_to_move = 0
        for file_to_move in all_files_to_move:
            moved_count, attempted_count = move_file_with_pkl(file_to_move, bad_folder, dry_run=True)
            total_files_to_move += attempted_count
        print_dry_run_summary(total_files_to_move, "move files to bad folder")
    
    if not all_files_to_move:
        print_summary(f"No files need to be moved - all supported images meet the {face_description} requirement!")

def main():
    parser = argparse.ArgumentParser(description='Move images that do not meet face count requirements to a bad subfolder')
    
    # Add standard training/testing arguments
    add_training_testing_args(parser)
    
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be moved without actually moving files')
    
    args = parser.parse_args()
    
    # Determine celebrity folder path and mode based on arguments
    mode, celebrity_name, celebrity_folder_path = get_mode_and_path_from_args(args)
    
    try:
        remove_bad_images(celebrity_folder_path, mode, args.dry_run)
        
    except Exception as e:
        print_error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()