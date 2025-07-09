#!/usr/bin/env python3
"""
StarMapr Headshot Detection Script

Automated script that takes a video URL and a list of celebrities, runs celebrity 
training for each, then downloads the video and extracts headshots for all successfully 
trained celebrities. Uses adaptive frame extraction if no headshots are initially found.

Usage:
    python3 run_headshot_detection.py "https://youtube.com/watch?v=VIDEO_ID" --show "SNL" "Bill Murray" "Tina Fey" "Amy Poehler"
    python3 run_headshot_detection.py "https://youtube.com/watch?v=VIDEO_ID" --show "SNL" --celebrities "Bill Murray,Tina Fey,Amy Poehler"
"""

import os
import sys
import subprocess
import argparse
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from utils import (
    get_celebrity_folder_name, get_env_int,
    print_error
)

# Load environment variables
load_dotenv()


def print_header(text):
    """Print a header in green color."""
    green = '\033[92m'
    reset = '\033[0m'
    print(f"{green}{text}{reset}")

def parse_celebrities(celebrity_args, celebrity_list_arg):
    """
    Parse celebrity names from either individual arguments or comma-separated list.
    
    Args:
        celebrity_args (list): Individual celebrity names as arguments
        celebrity_list_arg (str): Comma-separated celebrity names
        
    Returns:
        list: List of celebrity names
    """
    celebrities = []
    
    if celebrity_list_arg:
        celebrities.extend([name.strip() for name in celebrity_list_arg.split(',')])
    
    if celebrity_args:
        celebrities.extend(celebrity_args)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_celebrities = []
    for celeb in celebrities:
        if celeb not in seen:
            seen.add(celeb)
            unique_celebrities.append(celeb)
    
    return unique_celebrities


def run_subprocess_command(command_list, description, capture_output=False):
    """
    Run a subprocess command with error handling.
    
    Args:
        command_list (list): Command and arguments to run
        description (str): Description of the command for error reporting
        capture_output (bool): Whether to capture stdout/stderr
        
    Returns:
        tuple: (success: bool, stdout: str, stderr: str)
    """
    try:
        print(f"Running: {description}")
        result = subprocess.run(command_list, check=True, capture_output=capture_output, text=True)
        
        stdout = result.stdout if capture_output else ""
        stderr = result.stderr if capture_output else ""
        return True, stdout, stderr
    except subprocess.CalledProcessError as e:
        print_error(f"Failed: {description}")
        stdout = e.stdout if capture_output and e.stdout else ""
        stderr = e.stderr if capture_output and e.stderr else ""
        if stderr:
            print_error(stderr.strip())
        return False, stdout, stderr


def extract_video_folder_from_output(stdout, stderr):
    """
    Extract video folder path from download_video.py JSON output.
    
    Args:
        stdout (str): Standard output from download command (JSON format)
        stderr (str): Standard error from download command
        
    Returns:
        str or None: Video folder path if found
    """
    try:
        # Parse JSON output from download_video.py (last line)
        import json
        # Get the last non-empty line which should be the JSON output
        stdout_lines = stdout.strip().split('\n')
        last_line = stdout_lines[-1] if stdout_lines else ""
        result = json.loads(last_line)
        if result.get("success") and result.get("video_folder"):
            return result["video_folder"]
    except (json.JSONDecodeError, KeyError, AttributeError):
        # Fallback: try the old regex approach for backward compatibility
        combined_output = stdout + stderr
        pattern = r'05_videos/[a-zA-Z0-9_]+_[a-zA-Z0-9_-]+/?'
        matches = re.findall(pattern, combined_output)
        if matches:
            return matches[0].rstrip('/')
    
    return None


def run_celebrity_training(celebrity_name, show_name):
    """
    Run celebrity training pipeline for a single celebrity.
    
    Args:
        celebrity_name (str): Name of the celebrity
        show_name (str): Show name (required)
        
    Returns:
        bool: True if training successful, False otherwise
    """
    print_header(f"\n=== TRAINING: {celebrity_name} ===")
    
    command = ['python3', 'run_celebrity_training.py', celebrity_name, show_name]
    
    success, _, _ = run_subprocess_command(command, f"Training {celebrity_name}")
    
    if success:
        print(f"✓ Training completed for {celebrity_name}")
    else:
        print_error(f"✗ Training failed for {celebrity_name}")
    
    return success


def download_video(video_url):
    """
    Download video and return the folder path.
    
    Args:
        video_url (str): URL of the video to download
        
    Returns:
        str or None: Video folder path if successful, None if failed
    """
    print_header(f"\n=== DOWNLOADING VIDEO ===")
    print(f"URL: {video_url}")
    
    command = ['python3', 'download_video.py', video_url]
    success, stdout, stderr = run_subprocess_command(command, "Downloading video", capture_output=True)
    
    if success:
        video_folder = extract_video_folder_from_output(stdout, stderr)
        if video_folder:
            print(f"✓ Video downloaded to: {video_folder}")
            return video_folder
        else:
            print_error("Could not determine video folder path from output")
            return None
    else:
        print_error("✗ Video download failed")
        return None


def extract_frames_from_video(video_folder, frame_count):
    """
    Extract frames from video.
    
    Args:
        video_folder (str): Path to video folder
        frame_count (int): Number of frames to extract
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Extracting {frame_count} frames from video...")
    
    command = ['python3', 'extract_video_frames.py', video_folder, str(frame_count)]
    success, _, _ = run_subprocess_command(command, f"Extracting {frame_count} frames")
    
    return success


def extract_faces_from_frames(video_folder):
    """
    Extract faces from video frames.
    
    Args:
        video_folder (str): Path to video folder
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("Extracting faces from frames...")
    
    command = ['python3', 'extract_frame_faces.py', video_folder]
    success, _, _ = run_subprocess_command(command, "Extracting faces from frames")
    
    return success


def extract_celebrity_headshots(celebrity_name, video_folder):
    """
    Extract headshots for a specific celebrity from video.
    
    Args:
        celebrity_name (str): Name of the celebrity
        video_folder (str): Path to video folder
        
    Returns:
        tuple: (success: bool, headshot_count: int)
    """
    print(f"Extracting headshots for {celebrity_name}...")
    
    command = ['python3', 'extract_video_headshots.py', celebrity_name, video_folder]
    success, _, _ = run_subprocess_command(command, f"Extracting {celebrity_name} headshots")
    
    if success:
        # Count headshots in the celebrity-specific folder
        celeb_name_clean = celebrity_name.lower().replace(' ', '_')
        headshots_folder = Path(video_folder) / 'headshots' / celeb_name_clean
        if headshots_folder.exists():
            headshot_files = list(headshots_folder.glob('*.jpg')) + list(headshots_folder.glob('*.png'))
            return True, len(headshot_files)
    
    return False, 0


def run_operations_pipeline_with_adaptive_frames(video_folder, trained_celebrities):
    """
    Run the operations pipeline with adaptive frame extraction.
    
    Args:
        video_folder (str): Path to video folder
        trained_celebrities (list): List of successfully trained celebrity names
        
    Returns:
        dict: Dictionary mapping celebrity names to headshot counts
    """
    print_header(f"\n=== OPERATIONS PIPELINE ===")
    
    # Get default frame count
    default_frame_count = get_env_int('OPERATIONS_EXTRACT_FRAME_COUNT', 50)
    max_multiplier = 5
    results = {}
    
    for multiplier in range(1, max_multiplier + 1):
        current_frame_count = default_frame_count * multiplier
        
        if multiplier > 1:
            print(f"\n--- Attempt {multiplier}: {current_frame_count} frames ---")
        else:
            print(f"\n--- Initial attempt: {current_frame_count} frames ---")
        
        # Extract frames
        if not extract_frames_from_video(video_folder, current_frame_count):
            print_error("Failed to extract frames, aborting operations pipeline")
            break
        
        # Extract faces from frames
        if not extract_faces_from_frames(video_folder):
            print_error("Failed to extract faces from frames, aborting operations pipeline")
            break
        
        # Extract headshots for each trained celebrity
        total_headshots_found = 0
        for celebrity_name in trained_celebrities:
            success, headshot_count = extract_celebrity_headshots(celebrity_name, video_folder)
            if success:
                results[celebrity_name] = headshot_count
                total_headshots_found += headshot_count
                if headshot_count > 0:
                    print(f"✓ Found {headshot_count} headshots for {celebrity_name}")
                else:
                    print(f"No headshots found for {celebrity_name}")
            else:
                results[celebrity_name] = 0
                print_error(f"Failed to extract headshots for {celebrity_name}")
        
        # Check if all celebrities have at least 1 headshot
        celebrities_with_headshots = sum(1 for count in results.values() if count > 0)
        total_trained_celebrities = len(trained_celebrities)
        
        if celebrities_with_headshots == total_trained_celebrities or multiplier >= max_multiplier:
            if celebrities_with_headshots == total_trained_celebrities:
                print(f"✓ Found headshots for all {total_trained_celebrities} celebrities with {current_frame_count} frames")
            else:
                print(f"Found headshots for {celebrities_with_headshots}/{total_trained_celebrities} celebrities even with {current_frame_count} frames")
            break
        else:
            print(f"Found headshots for {celebrities_with_headshots}/{total_trained_celebrities} celebrities with {current_frame_count} frames, trying {default_frame_count * (multiplier + 1)} frames...")
    
    return results


def main():
    """Main function to run headshot detection pipeline."""
    start_time = time.time()
    
    parser = argparse.ArgumentParser(
        description='Run headshot detection pipeline for video and celebrities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://youtube.com/watch?v=ABC123" --show "SNL" --celebrities "Bill Murray,Tina Fey,Amy Poehler"
        """
    )
    
    parser.add_argument('video_url', help='URL of the video to download and process')
    parser.add_argument('celebrities', nargs='*', help='Celebrity names (space-separated)')
    parser.add_argument('--celebrities', dest='celebrity_list', 
                       help='Celebrity names (comma-separated)')
    parser.add_argument('--show', required=True,
                       help='Show/movie name for celebrity training (required)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show all output from subprocess commands')
    
    args = parser.parse_args()

    if args.verbose:
        os.environ['STARMAPR_LOG_VERBOSE'] = 'true'
    
    # Parse celebrity names
    celebrities = parse_celebrities(args.celebrities, args.celebrity_list)
    
    if not celebrities:
        print_error("No celebrities specified. Use either positional arguments or --celebrities flag.")
        sys.exit(1)
    
    print_header(f"=== HEADSHOT DETECTION PIPELINE ===")
    print(f"Video URL: {args.video_url}")
    print(f"Celebrities: {', '.join(celebrities)}")
    print(f"Show: {args.show}")
    
    # Step 1: Run celebrity training for each celebrity
    trained_celebrities = []
    failed_celebrities = []
    
    for celebrity_name in celebrities:
        if run_celebrity_training(celebrity_name, args.show):
            trained_celebrities.append(celebrity_name)
        else:
            failed_celebrities.append(celebrity_name)
    
    if not trained_celebrities:
        print_error("No celebrities were successfully trained. Aborting pipeline.")
        sys.exit(1)
    
    print_header(f"\nTraining Results:")
    print(f"✓ Successfully trained: {', '.join(trained_celebrities)}")
    if failed_celebrities:
        print(f"✗ Failed to train: {', '.join(failed_celebrities)}")
    
    # Step 2: Download video
    video_folder = download_video(args.video_url)
    if not video_folder:
        print_error("Video download failed. Aborting pipeline.")
        sys.exit(1)
    
    # Step 3: Run operations pipeline with adaptive frame extraction
    headshot_results = run_operations_pipeline_with_adaptive_frames(video_folder, trained_celebrities)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time / 60
    
    # Final summary
    print_header(f"\n=== FINAL RESULTS ===")
    print(f"Video folder: {video_folder}")
    print(f"Total execution time: {elapsed_minutes:.1f} minutes ({elapsed_time:.1f} seconds)")
    
    total_headshots = 0
    for celebrity_name in trained_celebrities:
        headshot_count = headshot_results.get(celebrity_name, 0)
        total_headshots += headshot_count
        if headshot_count > 0:
            print(f"✓ {celebrity_name}: {headshot_count} headshots")
        else:
            print_error(f"✗ {celebrity_name}: No headshots found")
    
    if failed_celebrities:
        print_error(f"Training failed: {', '.join(failed_celebrities)}")
    
    if total_headshots > 0:
        print(f"🎉 SUCCESS! Found {total_headshots} total headshots across all celebrities")
        sys.exit(0)
    else:
        print_error(f"❌ No headshots found for any celebrity")
        sys.exit(1)


if __name__ == '__main__':
    main()