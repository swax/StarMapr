#!/usr/bin/env python3
import os
import sys
import argparse
from deepface import DeepFace
from pathlib import Path
import shutil
import cv2
import numpy as np
from collections import defaultdict

def count_faces_in_image(image_path):
    """
    Count the number of faces detected in an image using DeepFace.
    
    Args:
        image_path (Path): Path to the image file
        
    Returns:
        int: Number of faces detected, -1 if error occurred
    """
    try:
        # Use DeepFace to detect faces (enforce_detection=False to avoid exceptions)
        face_analysis = DeepFace.represent(str(image_path), model_name='ArcFace', enforce_detection=False)
        return len(face_analysis) if face_analysis else 0
    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        return -1

def remove_bad_training_images(celebrity_folder_path, dry_run=False):
    """
    Move images from celebrity folder that don't have exactly 1 face detected to a 'bad' subfolder.
    
    Args:
        celebrity_folder_path (str): Path to celebrity folder containing training images
        dry_run (bool): If True, only report what would be moved without actually moving files
    """
    folder_path = Path(celebrity_folder_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Create bad folder if it doesn't exist
    bad_folder = folder_path / "bad"
    
    # Get all image files (excluding those already in bad folder)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in folder_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images in {folder_path}")
    if dry_run:
        print("DRY RUN MODE - No files will be moved")
    print()
    
    images_to_move = []
    images_with_one_face = []
    images_with_errors = []
    
    for img_file in image_files:
        print(f"Analyzing: {img_file.name}")
        face_count = count_faces_in_image(img_file)
        
        if face_count == -1:
            print(f"  → ERROR: Could not process image")
            images_with_errors.append(img_file)
        elif face_count == 0:
            print(f"  → MOVE TO BAD: No faces detected")
            images_to_move.append(img_file)
        elif face_count == 1:
            print(f"  → KEEP: Exactly 1 face detected")
            images_with_one_face.append(img_file)
        else:
            print(f"  → MOVE TO BAD: {face_count} faces detected (expected exactly 1)")
            images_to_move.append(img_file)
    
    # Summary
    print(f"\nSummary:")
    print(f"Total images analyzed: {len(image_files)}")
    print(f"Images with exactly 1 face (keeping): {len(images_with_one_face)}")
    print(f"Images to move to bad folder: {len(images_to_move)}")
    print(f"Images with processing errors: {len(images_with_errors)}")
    
    if images_to_move:
        print(f"\nImages to be moved to bad folder:")
        for img in images_to_move:
            print(f"  - {img.name}")
    
    if images_with_errors:
        print(f"\nImages with processing errors (not moved):")
        for img in images_with_errors:
            print(f"  - {img.name}")
    
    # Move bad images if not in dry run mode
    if images_to_move and not dry_run:
        # Create bad folder
        bad_folder.mkdir(exist_ok=True)
        print(f"\nMoving {len(images_to_move)} images to {bad_folder}...")
        
        moved_count = 0
        for img_file in images_to_move:
            try:
                destination = bad_folder / img_file.name
                shutil.move(str(img_file), str(destination))
                print(f"  ✓ Moved: {img_file.name}")
                moved_count += 1
            except Exception as e:
                print(f"  ✗ Failed to move {img_file.name}: {e}")
        
        print(f"\nSuccessfully moved {moved_count}/{len(images_to_move)} images to bad folder")
    
    elif images_to_move and dry_run:
        print(f"\nDRY RUN: Would move {len(images_to_move)} images to bad folder")
    
    if not images_to_move:
        print(f"\nNo images need to be moved - all images have exactly 1 face!")

def main():
    parser = argparse.ArgumentParser(description='Move training images that do not have exactly 1 face detected to a bad subfolder')
    parser.add_argument('celebrity_folder', help='Path to celebrity folder containing training images')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be moved without actually moving files')
    
    args = parser.parse_args()
    
    try:
        remove_bad_training_images(args.celebrity_folder, args.dry_run)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()