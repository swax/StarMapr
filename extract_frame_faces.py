#!/usr/bin/env python3

import argparse
from pathlib import Path
from utils import get_image_files, print_dry_run_header, print_error, print_summary, log
from utils_deepface import get_face_embeddings


def main():
    parser = argparse.ArgumentParser(description='Detect faces in extracted frames and save bounding box data')
    parser.add_argument('video_folder', help='Path to video folder (frames subfolder will be used)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without actually doing it')
    
    args = parser.parse_args()
    
    # Construct frames directory path
    video_folder = Path(args.video_folder)
    frames_dir = video_folder / "frames"
    
    # Validate video folder
    if not video_folder.exists():
        print_error(f"Video folder not found: {video_folder}")
        sys.exit(1)
    
    # Validate frames directory
    if not frames_dir.exists():
        print_error(f"Frames directory not found: {frames_dir}")
        print_error("Make sure you've extracted frames first using extract_video_frames.py")
        sys.exit(1)
    
    # Get all image files
    image_files = get_image_files(frames_dir, exclude_subdirs=True)
    
    if not image_files:
        print_error(f"No image files found in {frames_dir}")
        sys.exit(1)
    
    # Sort by filename to process in order
    image_files.sort()
    
    log(f"Found {len(image_files)} frame images")
    log(f"Face data will be saved to: {frames_dir}")
    
    if args.dry_run:
        print_dry_run_header("no files will be created")
        for img_file in image_files:
            pkl_path = img_file.with_suffix('.pkl')
            status = "EXISTS" if pkl_path.exists() else "WOULD CREATE"
            log(f"  {img_file.name} -> {pkl_path.name} [{status}]")
        sys.exit(0)
    
    # Process frames
    processed_count = 0
    skipped_count = 0
    total_faces = 0
    
    for img_file in image_files:
        pkl_path = img_file.with_suffix('.pkl')
        
        # Check if face data already exists (caching is now handled in the utility function)
        if pkl_path.exists():
            log(f"Skipping {img_file.name} - face data already exists")
            skipped_count += 1
            continue
        
        log(f"Processing {img_file.name}...")
        
        try:
            # Detect faces (automatically handles caching)
            faces_data = get_face_embeddings(img_file, headshotable_only=True)
            if faces_data is None:
                faces_data = []
            
            face_count = len(faces_data)
            total_faces += face_count
            processed_count += 1
            log(f"  → Detected {face_count} faces -> {pkl_path.name}")
                
        except Exception as e:
            print_error(f"Error processing {img_file.name}: {str(e)}")
    
    log(f"\nSummary:")
    log(f"  Frames processed: {processed_count}")
    log(f"  Frames skipped: {skipped_count}")
    log(f"  Total faces detected: {total_faces}")
    
    if processed_count > 0:
        print_summary(f"Face detection completed! Processed {processed_count} frames and detected {total_faces} faces.")
    elif skipped_count > 0:
        print_summary(f"All {skipped_count} frames already processed - no face detection needed.")
    else:
        print_error("No frames were processed.")

if __name__ == "__main__":
    main()