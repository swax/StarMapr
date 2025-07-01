#!/usr/bin/env python3

import cv2
import os
import argparse
from pathlib import Path

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

def main():
    parser = argparse.ArgumentParser(description='Extract frames from video using binary search pattern')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('num_frames', type=int, help='Number of frames to extract')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without actually doing it')
    
    args = parser.parse_args()
    
    # Validate video file
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    # Create frames directory in same location as video
    video_path = Path(args.video_path)
    frames_dir = video_path.parent / "frames"
    
    # Get video info
    try:
        total_frames = get_video_frame_count(args.video_path)
        print(f"Video has {total_frames} total frames")
    except Exception as e:
        print(f"Error reading video: {str(e)}")
        return 1
    
    # Generate frame indices using binary search pattern
    frame_indices = generate_binary_search_indices(total_frames, args.num_frames)
    print(f"Will extract {len(frame_indices)} frames using binary search pattern")
    print(f"Frame indices: {frame_indices}")
    print(f"Frames will be saved to: {frames_dir}")
    
    if args.dry_run:
        print("Dry run - no files will be created")
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
            print(f"Skipping frame {frame_idx} - already exists: {frame_path}")
            skipped_count += 1
            continue
        
        try:
            # Extract frame
            extract_frame(args.video_path, frame_idx, str(frame_path))
            print(f"Extracted frame {frame_idx} -> {frame_path}")
            extracted_count += 1
                
        except Exception as e:
            print(f"Error extracting frame {frame_idx}: {str(e)}")
    
    print(f"\nSummary:")
    print(f"  Frames extracted: {extracted_count}")
    print(f"  Frames skipped: {skipped_count}")

if __name__ == "__main__":
    exit(main())