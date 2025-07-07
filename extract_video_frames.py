#!/usr/bin/env python3

import cv2
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from utils import get_env_int, print_dry_run_header, print_error, print_summary, log

# Load environment variables
load_dotenv()

def get_video_frame_count(video_path):
    """Get total number of frames in video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def generate_binary_search_indices(total_frames, num_frames):
    """Generate frame indices using binary search pattern starting from middle"""
    if num_frames >= total_frames:
        return list(range(total_frames))
    
    indices = set()
    
    # Start with middle frame
    queue = [(0, total_frames - 1)]
    
    while len(indices) < num_frames and queue:
        start, end = queue.pop(0)
        
        if start > end:
            continue
            
        # Calculate middle point
        mid = (start + end) // 2
        
        if mid not in indices:
            indices.add(mid)
            
            if len(indices) >= num_frames:
                break
        
        # Add left and right halves to queue if we need more frames
        if len(indices) < num_frames:
            if mid > start:
                queue.append((start, mid - 1))
            if mid < end:
                queue.append((mid + 1, end))
    
    return sorted(list(indices))

def extract_frame(video_path, frame_number, output_path):
    """Extract a specific frame from video and save as image"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Cannot read frame {frame_number} from video")
    
    # Save frame
    cv2.imwrite(output_path, frame)
    return True

def find_video_file(folder_path):
    """Find video file in the given folder"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}
    
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder not found: {folder_path}")
    
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    video_files = []
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in video_extensions:
            video_files.append(file)
    
    if not video_files:
        raise ValueError(f"No video files found in folder: {folder_path}")
    
    if len(video_files) > 1:
        log(f"Multiple video files found, using: {video_files[0].name}")
    
    return video_files[0]

def main():
    parser = argparse.ArgumentParser(description='Extract frames from video using binary search pattern')
    parser.add_argument('folder_path', help='Path to folder containing video file')
    
    # Get default frame count from environment variable
    default_frame_count = get_env_int('OPERATIONS_EXTRACT_FRAME_COUNT', 50)
    parser.add_argument('num_frames', type=int, nargs='?', default=default_frame_count,
                       help=f'Number of frames to extract (default: {default_frame_count})')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without actually doing it')
    
    args = parser.parse_args()
    
    # Find video file in folder
    try:
        video_file = find_video_file(args.folder_path)
        log(f"Found video file: {video_file}")
    except Exception as e:
        print_error(str(e))
        return 1
    
    # Create frames directory in same folder
    folder_path = Path(args.folder_path)
    frames_dir = folder_path / "frames"
    
    # Get video info
    try:
        total_frames = get_video_frame_count(str(video_file))
        log(f"Video has {total_frames} total frames")
    except Exception as e:
        print_error(f"Error reading video: {str(e)}")
        return 1
    
    # Generate frame indices using binary search pattern
    frame_indices = generate_binary_search_indices(total_frames, args.num_frames)
    log(f"Will extract {len(frame_indices)} frames using binary search pattern")
    log(f"Frame indices: {frame_indices}")
    log(f"Frames will be saved to: {frames_dir}")
    
    if args.dry_run:
        print_dry_run_header("no files will be created")
        return 0
    
    # Create frames directory
    frames_dir.mkdir(exist_ok=True)
    
    # Extract frames
    extracted_count = 0
    skipped_count = 0
    
    for frame_idx in frame_indices:
        # Name file with actual frame number, zero-padded
        frame_filename = f"{frame_idx:08d}.jpg"
        frame_path = frames_dir / frame_filename
        
        # Skip if frame already exists
        if frame_path.exists():
            log(f"Skipping frame {frame_idx} - already exists: {frame_path}")
            skipped_count += 1
            continue
        
        try:
            # Extract frame
            extract_frame(str(video_file), frame_idx, str(frame_path))
            log(f"Extracted frame {frame_idx} -> {frame_path}")
            extracted_count += 1
                
        except Exception as e:
            print_error(f"Error extracting frame {frame_idx}: {str(e)}")
    
    log(f"\nSummary:")
    log(f"  Frames extracted: {extracted_count}")
    log(f"  Frames skipped: {skipped_count}")
    
    if extracted_count > 0:
        print_summary(f"Frame extraction completed! Extracted {extracted_count} frames to {frames_dir}")
    elif skipped_count > 0:
        print_summary(f"All {skipped_count} frames already existed - no extraction needed.")
    else:
        print_error("No frames were extracted.")

if __name__ == "__main__":
    exit(main())