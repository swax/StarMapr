#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np
from collections import defaultdict
from utils import add_training_testing_args, get_mode_and_path_from_args, print_dry_run_header, print_dry_run_summary, get_supported_image_extensions, print_error, print_summary, move_file_with_pkl, log
from utils_deepface import get_face_embeddings

def count_faces_in_image(image_path):
    """
    Count the number of faces detected in an image using DeepFace.
    
    Args:
        image_path (Path): Path to the image file
        
    Returns:
        int: Number of faces detected, -1 if error occurred
    """
    face_analysis = get_face_embeddings(image_path)
    if face_analysis is None:
        return -1  # Error occurred
    return len(face_analysis)  # 0 or more faces detected

def move_files_to_folder(files_to_move, destination_folder, folder_name, dry_run=False):
    """
    Helper function to move files to a specific folder and return move counts.
    
    Args:
        files_to_move (list): List of file paths to move
        destination_folder (Path): Destination folder path
        folder_name (str): Name of the folder for logging
        dry_run (bool): If True, only simulate the moves
        
    Returns:
        tuple: (total_moved_count, total_attempted_count)
    """
    moved_count = 0
    attempted_count = 0
    
    if files_to_move:
        if not dry_run:
            log(f"\nMoving {len(files_to_move)} files to {destination_folder}...")
        
        for file_to_move in files_to_move:
            file_moved_count, file_attempted_count = move_file_with_pkl(file_to_move, destination_folder, dry_run)
            moved_count += file_moved_count
            attempted_count += file_attempted_count
    
    return moved_count, attempted_count

def remove_bad_images(actor_folder_path, mode='training', dry_run=False):
    """
    Move images from actor folder that don't meet face count requirements to categorized subfolders.
    
    Args:
        actor_folder_path (str): Path to actor folder containing training or testing images
        mode (str): 'training' expects exactly 1 face, 'testing' expects 4-10 faces
        dry_run (bool): If True, only report what would be moved without actually moving files
    """
    folder_path = Path(actor_folder_path)
    
    if not folder_path.exists():
        print_error(f"Folder not found: {folder_path}")
        return
    
    # Create categorized folders for different types of bad images
    bad_face_count_folder = folder_path / "bad_face_count"
    unsupported_folder = folder_path / "bad_unsupported"
    error_folder = folder_path / "bad_error"
    
    # Get all files (excluding those already in categorized folders and subdirectories)
    image_extensions = get_supported_image_extensions()
    all_files = [f for f in folder_path.iterdir() 
                 if f.is_file() and not f.name.startswith('.')]
    
    # Separate supported and unsupported files
    image_files = [f for f in all_files if f.suffix.lower() in image_extensions]
    unsupported_files = [f for f in all_files if f.suffix.lower() not in image_extensions]
    
    if not all_files:
        print_error(f"No files found in {folder_path}")
        return
    
    log(f"Found {len(all_files)} files in {folder_path}")
    log(f"  - {len(image_files)} supported image files")
    log(f"  - {len(unsupported_files)} unsupported format files")
    if dry_run:
        print_dry_run_header("No files will be moved")
    log()
    
    images_to_move = []
    images_with_good_faces = []
    images_with_errors = []
    unsupported_to_move = [f for f in unsupported_files if f.suffix.lower() != '.pkl']
    
    # Define face count requirements based on mode
    if mode == 'training':
        required_faces = 1
        face_description = "exactly 1 face"
    else:  # testing mode
        min_faces, max_faces = 3, 10
        face_description = f"{min_faces}-{max_faces} faces"
    
    for img_file in image_files:
        log(f"Analyzing: {img_file.name}")
        face_count = count_faces_in_image(img_file)
        
        if face_count == -1:
            log(f"  → MOVE TO ERROR: Could not process image")
            images_with_errors.append(img_file)
        elif mode == 'training':
            if face_count == required_faces:
                log(f"  → KEEP: Exactly 1 face detected")
                images_with_good_faces.append(img_file)
            else:
                if face_count == 0:
                    log(f"  → MOVE TO BAD_FACE_COUNT: No faces detected")
                else:
                    log(f"  → MOVE TO BAD_FACE_COUNT: {face_count} faces detected (expected exactly 1)")
                images_to_move.append(img_file)
        else:  # testing mode
            if min_faces <= face_count <= max_faces:
                log(f"  → KEEP: {face_count} faces detected (within {min_faces}-{max_faces} range)")
                images_with_good_faces.append(img_file)
            else:
                if face_count == 0:
                    log(f"  → MOVE TO BAD_FACE_COUNT: No faces detected")
                elif face_count < min_faces:
                    log(f"  → MOVE TO BAD_FACE_COUNT: {face_count} faces detected (need at least {min_faces})")
                else:
                    log(f"  → MOVE TO BAD_FACE_COUNT: {face_count} faces detected (maximum {max_faces} allowed)")
                images_to_move.append(img_file)
    
    # Handle unsupported files
    if unsupported_to_move:
        log(f"\nUnsupported format files (will be moved to unsupported folder):")
        for file in unsupported_to_move:
            log(f"  → MOVE TO UNSUPPORTED: {file.name} (unsupported format: {file.suffix or 'no extension'})")
    
    # Summary
    log(f"\nSummary:")
    log(f"Total files found: {len(all_files)}")
    log(f"Supported images analyzed: {len(image_files)}")
    log(f"Images with {face_description} (keeping): {len(images_with_good_faces)}")
    log(f"Images to move to bad_face_count folder (wrong face count): {len(images_to_move)}")
    log(f"Unsupported files to move to unsupported folder: {len(unsupported_to_move)}")
    log(f"Images with processing errors to move to error folder: {len(images_with_errors)}")
    
    # Move files to their respective categorized folders
    total_moved_count = 0
    total_attempted_count = 0
    
    # Move bad face images
    moved_count, attempted_count = move_files_to_folder(images_to_move, bad_face_count_folder, "bad_face_count", dry_run)
    total_moved_count += moved_count
    total_attempted_count += attempted_count
    
    # Move unsupported files
    moved_count, attempted_count = move_files_to_folder(unsupported_to_move, unsupported_folder, "unsupported", dry_run)
    total_moved_count += moved_count
    total_attempted_count += attempted_count
    
    # Move error files
    moved_count, attempted_count = move_files_to_folder(images_with_errors, error_folder, "error", dry_run)
    total_moved_count += moved_count
    total_attempted_count += attempted_count
    
    # Print summary
    if total_attempted_count > 0:
        if dry_run:
            print_dry_run_summary(total_attempted_count, "move files to categorized folders")
        else:
            print_summary(f"Successfully moved {total_moved_count}/{total_attempted_count} files to categorized folders")
    else:
        print_summary(f"No files need to be moved - all supported images meet the {face_description} requirement!")

def main():
    parser = argparse.ArgumentParser(description='Move images that do not meet face count requirements to a bad subfolder')
    
    # Add standard training/testing arguments
    add_training_testing_args(parser)
    
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be moved without actually moving files')
    
    args = parser.parse_args()
    
    # Determine actor folder path and mode based on arguments
    mode, actor_name, actor_folder_path = get_mode_and_path_from_args(args)
    
    try:
        remove_bad_images(actor_folder_path, mode, args.dry_run)
        
    except Exception as e:
        print_error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()