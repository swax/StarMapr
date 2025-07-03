#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv
from utils import add_training_testing_args, get_mode_and_path_from_args, get_image_files, get_env_int, print_dry_run_header, print_dry_run_summary, print_error, print_summary

# Load environment variables
load_dotenv()

def compute_perceptual_hash(image_path, hash_size=8):
    """
    Compute perceptual hash for an image using average hash algorithm.
    
    Args:
        image_path (Path): Path to the image file
        hash_size (int): Size of the hash matrix (default 8x8)
        
    Returns:
        str: Hexadecimal hash string, or None if error occurred
    """
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to hash_size x hash_size
        resized = cv2.resize(gray, (hash_size, hash_size))
        
        # Calculate average
        avg = resized.mean()
        
        # Create hash: 1 if pixel > average, 0 otherwise
        hash_bits = resized > avg
        
        # Convert to hex string
        hash_bytes = np.packbits(hash_bits.flatten())
        return hash_bytes.tobytes().hex()
        
    except Exception as e:
        print_error(f"Error computing hash for {image_path.name}: {e}")
        return None

def hamming_distance(hash1, hash2):
    """
    Calculate Hamming distance between two hash strings.
    
    Args:
        hash1, hash2 (str): Hexadecimal hash strings
        
    Returns:
        int: Hamming distance (number of different bits)
    """
    if len(hash1) != len(hash2):
        return float('inf')
    
    # Convert hex to binary and count differences
    distance = 0
    for i in range(0, len(hash1), 2):
        byte1 = int(hash1[i:i+2], 16)
        byte2 = int(hash2[i:i+2], 16)
        distance += bin(byte1 ^ byte2).count('1')
    
    return distance

def find_duplicate_groups(image_files, similarity_threshold=5):
    """
    Find groups of near-duplicate images using perceptual hashing.
    
    Args:
        image_files (list): List of image file paths
        similarity_threshold (int): Maximum Hamming distance for duplicates
        
    Returns:
        list: List of duplicate groups, each group is a list of similar images
    """
    print(f"Computing perceptual hashes for duplicate detection...")
    
    # Compute hashes for all images
    image_hashes = {}
    for img_file in image_files:
        hash_value = compute_perceptual_hash(img_file)
        if hash_value:
            image_hashes[img_file] = hash_value
    
    print(f"Successfully computed hashes for {len(image_hashes)}/{len(image_files)} images")
    
    # Find duplicate groups
    processed = set()
    duplicate_groups = []
    
    for img1, hash1 in image_hashes.items():
        if img1 in processed:
            continue
            
        # Find all images similar to img1
        similar_group = [img1]
        processed.add(img1)
        
        for img2, hash2 in image_hashes.items():
            if img2 in processed:
                continue
                
            distance = hamming_distance(hash1, hash2)
            if distance <= similarity_threshold:
                similar_group.append(img2)
                processed.add(img2)
        
        # Only add groups with more than 1 image
        if len(similar_group) > 1:
            duplicate_groups.append(similar_group)
    
    return duplicate_groups

def remove_duplicate_images(celebrity_folder_path, similarity_threshold=5, dry_run=False):
    """
    Remove near-duplicate images from celebrity folder, keeping the oldest one in each group.
    Keeping the oldest is important as it may have a corresponding pkl file that we don't want to abandon.
    
    Args:
        celebrity_folder_path (str): Path to celebrity folder containing training or testing images
        similarity_threshold (int): Maximum Hamming distance for considering images duplicates
        dry_run (bool): If True, only report what would be moved without actually moving files
    """
    folder_path = Path(celebrity_folder_path)
    
    if not folder_path.exists():
        print_error(f"Folder not found: {folder_path}")
        return
    
    # Create duplicates folder if it doesn't exist
    duplicates_folder = folder_path / "duplicates"
    
    # Get all image files (excluding those already in duplicates folder)
    image_files = get_image_files(folder_path, exclude_subdirs=True)
    
    if not image_files:
        print_error(f"No image files found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images in {folder_path}")
    if dry_run:
        print_dry_run_header("No files will be moved")
    print(f"Using similarity threshold: {similarity_threshold} (lower = more strict)")
    print()
    
    # Find duplicate groups
    duplicate_groups = find_duplicate_groups(image_files, similarity_threshold)
    
    if not duplicate_groups:
        print("No duplicate images found!")
        return
    
    print(f"\nFound {len(duplicate_groups)} groups of duplicate images:")
    
    images_to_move = []
    
    for i, group in enumerate(duplicate_groups, 1):
        print(f"\nGroup {i} ({len(group)} similar images):")
        # Sort by modification time (keep oldest)
        group.sort(key=lambda x: x.stat().st_mtime)
        
        keep_image = group[0]
        duplicate_images = group[1:]
        
        print(f"  KEEP: {keep_image.name} ({keep_image.stat().st_size} bytes)")
        for img in duplicate_images:
            print(f"  MOVE: {img.name} ({img.stat().st_size} bytes)")
            images_to_move.append(img)
    
    # Summary
    print(f"\nSummary:")
    print(f"Total images analyzed: {len(image_files)}")
    print(f"Duplicate groups found: {len(duplicate_groups)}")
    print(f"Images to keep: {len(image_files) - len(images_to_move)}")
    print(f"Images to move to duplicates folder: {len(images_to_move)}")
    
    # Move duplicate images if not in dry run mode
    if images_to_move and not dry_run:
        # Create duplicates folder
        duplicates_folder.mkdir(exist_ok=True)
        print(f"\nMoving {len(images_to_move)} duplicate images to {duplicates_folder}...")
        
        moved_count = 0
        for img_file in images_to_move:
            try:
                destination = duplicates_folder / img_file.name
                # Handle name conflicts
                counter = 1
                while destination.exists():
                    stem = img_file.stem
                    suffix = img_file.suffix
                    destination = duplicates_folder / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                shutil.move(str(img_file), str(destination))
                print(f"  ✓ Moved: {img_file.name}")
                moved_count += 1
            except Exception as e:
                print_error(f"Failed to move {img_file.name}: {e}")
        
        print_summary(f"Successfully moved {moved_count}/{len(images_to_move)} duplicate images to duplicates folder")
    
    elif images_to_move and dry_run:
        print_dry_run_summary(len(images_to_move), "move duplicate images")

def main():
    parser = argparse.ArgumentParser(description='Remove near-duplicate images from celebrity folder')
    
    # Add standard training/testing arguments
    add_training_testing_args(parser)
    
    # Get default threshold from environment variable
    default_threshold = get_env_int('TRAINING_DUPLICATE_THRESHOLD', 5)
    parser.add_argument('--threshold', type=int, default=default_threshold,
                       help=f'Similarity threshold (0-64, lower = more strict, default: {default_threshold})')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be moved without actually moving files')
    
    args = parser.parse_args()
    
    if args.threshold < 0 or args.threshold > 64:
        print_error("Threshold must be between 0 and 64")
        sys.exit(1)
    
    # Determine celebrity folder path based on mode
    mode, celebrity_name, celebrity_folder_path = get_mode_and_path_from_args(args)
    
    try:
        remove_duplicate_images(celebrity_folder_path, args.threshold, args.dry_run)
        
    except Exception as e:
        print_error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()