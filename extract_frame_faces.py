#!/usr/bin/env python3

import os
import argparse
import pickle
from pathlib import Path
from deepface import DeepFace
from utilities import print_error, print_summary

def detect_faces_in_frame(frame_path):
    """Detect all faces in a frame and return face data with bounding boxes"""
    try:
        # Use DeepFace to detect faces and get facial areas
        face_analysis = DeepFace.represent(str(frame_path), model_name='ArcFace', enforce_detection=False)
        
        if not face_analysis:
            return []
        
        faces_data = []
        for i, face_data in enumerate(face_analysis):
            face_region = face_data['facial_area']
            faces_data.append({
                'face_id': i + 1,
                'bounding_box': {
                    'x': face_region['x'],
                    'y': face_region['y'],
                    'w': face_region['w'],
                    'h': face_region['h']
                },
                'embedding': face_data['embedding']
            })
        
        return faces_data
        
    except Exception as e:
        print_error(f"Error detecting faces in {frame_path}: {str(e)}")
        return []

def save_face_data(frame_path, faces_data):
    """Save face detection data to pickle file alongside frame"""
    pkl_path = frame_path.with_suffix('.pkl')
    
    frame_data = {
        'frame_file': frame_path.name,
        'total_faces': len(faces_data),
        'faces': faces_data
    }
    
    try:
        with open(pkl_path, 'wb') as f:
            pickle.dump(frame_data, f)
        return True
    except Exception as e:
        print_error(f"Error saving face data to {pkl_path}: {str(e)}")
        return False

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
        return 1
    
    # Validate frames directory
    if not frames_dir.exists():
        print_error(f"Frames directory not found: {frames_dir}")
        print_error("Make sure you've extracted frames first using extract_video_frames.py")
        return 1
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in frames_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print_error(f"No image files found in {frames_dir}")
        return 1
    
    # Sort by filename to process in order
    image_files.sort()
    
    print(f"Found {len(image_files)} frame images")
    print(f"Face data will be saved to: {frames_dir}")
    
    if args.dry_run:
        print("Dry run - no files will be created")
        for img_file in image_files:
            pkl_path = img_file.with_suffix('.pkl')
            status = "EXISTS" if pkl_path.exists() else "WOULD CREATE"
            print(f"  {img_file.name} -> {pkl_path.name} [{status}]")
        return 0
    
    # Process frames
    processed_count = 0
    skipped_count = 0
    total_faces = 0
    
    for img_file in image_files:
        pkl_path = img_file.with_suffix('.pkl')
        
        # Skip if face data already exists
        if pkl_path.exists():
            print(f"Skipping {img_file.name} - face data already exists")
            skipped_count += 1
            continue
        
        print(f"Processing {img_file.name}...")
        
        try:
            # Detect faces
            faces_data = detect_faces_in_frame(img_file)
            
            # Save face data
            if save_face_data(img_file, faces_data):
                face_count = len(faces_data)
                total_faces += face_count
                processed_count += 1
                print(f"  → Detected {face_count} faces -> {pkl_path.name}")
            else:
                print(f"  → Failed to save face data")
                
        except Exception as e:
            print_error(f"Error processing {img_file.name}: {str(e)}")
    
    print(f"\nSummary:")
    print(f"  Frames processed: {processed_count}")
    print(f"  Frames skipped: {skipped_count}")
    print(f"  Total faces detected: {total_faces}")
    
    if processed_count > 0:
        print_summary(f"Face detection completed! Processed {processed_count} frames and detected {total_faces} faces.")
    elif skipped_count > 0:
        print_summary(f"All {skipped_count} frames already processed - no face detection needed.")
    else:
        print_error("No frames were processed.")

if __name__ == "__main__":
    exit(main())