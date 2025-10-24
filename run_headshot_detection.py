#!/usr/bin/env python3
"""
StarMapr Headshot Detection Script

Automated script that takes a video URL and a list of actors, runs actor 
training for each, then downloads the video and extracts headshots for all successfully 
trained actors. Uses adaptive frame extraction if no headshots are initially found.

Usage:
    python3 run_headshot_detection.py "https://youtube.com/watch?v=VIDEO_ID" --show "SNL" "Bill Murray" "Tina Fey" "Amy Poehler"
    python3 run_headshot_detection.py "https://youtube.com/watch?v=VIDEO_ID" --show "SNL" --actors "Bill Murray,Tina Fey,Amy Poehler"
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
    log, get_actor_folder_name, get_env_int, print_error
)

# Load environment variables
load_dotenv()


def print_header(text):
    """Print a header in green color."""
    green = '\033[92m'
    reset = '\033[0m'
    print(f"{green}{text}{reset}")

def parse_actors(actor_args, actor_list_arg):
    """
    Parse actor names from either individual arguments or comma-separated list.
    
    Args:
        actor_args (list): Individual actor names as arguments
        actor_list_arg (str): Comma-separated actor names
        
    Returns:
        list: List of actor names
    """
    actors = []
    
    if actor_list_arg:
        actors.extend([name.strip() for name in actor_list_arg.split(',')])
    
    if actor_args:
        actors.extend(actor_args)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_actors = []
    for actor in actors:
        if actor not in seen:
            seen.add(actor)
            unique_actors.append(actor)
    
    return unique_actors


def run_subprocess_command(command_list, description):
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
        # Don't show real time output because there are unavoidable cuda errors that get piped to the console, filling the context
        result = subprocess.run(command_list, check=True, capture_output=True, text=True)
        
        # Show output from successful commands
        if result.stdout:
            log(result.stdout.strip())
        
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        print_error(f"Failed: {description}")
        if e.stderr:
            print_error(e.stderr.strip())
        elif e.stdout:
            print_error(e.stdout.strip())
        return False, e.stdout, e.stderr


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


def run_actor_training(actor_name, show_name):
    """
    Run actor training pipeline for a single actor.
    
    Args:
        actor_name (str): Name of the actor
        show_name (str): Show name (required)
        
    Returns:
        bool: True if training successful, False otherwise
    """
    print_header(f"\n=== TRAINING: {actor_name} ===")
    
    command = ['venv/bin/python3', 'run_actor_training.py', actor_name, show_name]
    
    success, _, _ = run_subprocess_command(command, f"Training {actor_name}")
    
    if success:
        print(f"✓ Training completed for {actor_name}")
    else:
        print_error(f"✗ Training failed for {actor_name}")
    
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
    
    command = ['venv/bin/python3', 'download_video.py', video_url]
    success, stdout, stderr = run_subprocess_command(command, "Downloading video")
    
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
    
    command = ['venv/bin/python3', 'extract_video_frames.py', video_folder, str(frame_count)]
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
    
    command = ['venv/bin/python3', 'extract_frame_faces.py', video_folder]
    success, _, _ = run_subprocess_command(command, "Extracting faces from frames")
    
    return success


def extract_actor_headshots(actor_name, video_folder):
    """
    Extract headshots for a specific actor from video.
    
    Args:
        actor_name (str): Name of the actor
        video_folder (str): Path to video folder
        
    Returns:
        tuple: (success: bool, headshot_count: int)
    """
    print(f"Extracting headshots for {actor_name}...")
    
    command = ['venv/bin/python3', 'extract_video_headshots.py', actor_name, video_folder]
    success, _, _ = run_subprocess_command(command, f"Extracting {actor_name} headshots")
    
    if success:
        # Count headshots in the actor-specific folder
        actor_name_clean = get_actor_folder_name(actor_name)
        headshots_folder = Path(video_folder) / 'headshots' / actor_name_clean
        if headshots_folder.exists():
            headshot_files = list(headshots_folder.glob('*.jpg')) + list(headshots_folder.glob('*.png'))
            return True, len(headshot_files)
    
    return False, 0


def run_operations_pipeline_with_adaptive_frames(video_folder, trained_actors):
    """
    Run the operations pipeline with adaptive frame extraction.
    
    Args:
        video_folder (str): Path to video folder
        trained_actors (list): List of successfully trained actor names
        
    Returns:
        dict: Dictionary mapping actor names to headshot counts
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
        
        # Extract headshots for each trained actor
        total_headshots_found = 0
        for actor_name in trained_actors:
            success, headshot_count = extract_actor_headshots(actor_name, video_folder)
            if success:
                results[actor_name] = headshot_count
                total_headshots_found += headshot_count
                if headshot_count > 0:
                    print(f"✓ Found {headshot_count} headshots for {actor_name}")
                else:
                    print(f"No headshots found for {actor_name}")
            else:
                results[actor_name] = 0
                print_error(f"Failed to extract headshots for {actor_name}")
        
        # Check if all actors have at least 1 headshot
        actors_with_headshots = sum(1 for count in results.values() if count > 0)
        total_trained_actors = len(trained_actors)
        
        if actors_with_headshots == total_trained_actors or multiplier >= max_multiplier:
            if actors_with_headshots == total_trained_actors:
                print(f"✓ Found headshots for all {total_trained_actors} actors with {current_frame_count} frames")
            else:
                print(f"Found headshots for {actors_with_headshots}/{total_trained_actors} actors even with {current_frame_count} frames")
            break
        else:
            print(f"Found headshots for {actors_with_headshots}/{total_trained_actors} actors with {current_frame_count} frames, trying {default_frame_count * (multiplier + 1)} frames...")
    
    return results


def main():
    """Main function to run headshot detection pipeline."""
    start_time = time.time()
    
    parser = argparse.ArgumentParser(
        description='Run headshot detection pipeline for video and actors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://youtube.com/watch?v=ABC123" --show "SNL" --actors "Bill Murray,Tina Fey,Amy Poehler"
        """
    )
    
    parser.add_argument('video_url', help='URL of the video to download and process')
    parser.add_argument('actors', nargs='*', help='Actor names (space-separated)')
    parser.add_argument('--actors', dest='actor_list', 
                       help='Actor names (comma-separated)')
    parser.add_argument('--show', required=True,
                       help='Show/movie name for actor training (required)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show all output from subprocess commands')
    
    args = parser.parse_args()

    if args.verbose:
        os.environ['STARMAPR_LOG_VERBOSE'] = 'true'
    
    # Parse actor names
    actors = parse_actors(args.actors, args.actor_list)
    
    if not actors:
        print_error("No actors specified. Use either positional arguments or --actors flag.")
        sys.exit(1)
    
    print_header(f"=== HEADSHOT DETECTION PIPELINE ===")
    print(f"Video URL: {args.video_url}")
    print(f"Actors: {', '.join(actors)}")
    print(f"Show: {args.show}")
    
    # Step 1: Run actor training for each actor
    trained_actors = []
    failed_actors = []
    
    for actor_name in actors:
        if run_actor_training(actor_name, args.show):
            trained_actors.append(actor_name)
        else:
            failed_actors.append(actor_name)
    
    if not trained_actors:
        print_error("No actors were successfully trained. Aborting pipeline.")
        sys.exit(1)
    
    print_header(f"\nTraining Results:")
    print(f"✓ Successfully trained: {', '.join(trained_actors)}")
    if failed_actors:
        print(f"✗ Failed to train: {', '.join(failed_actors)}")
    
    # Step 2: Download video
    video_folder = download_video(args.video_url)
    if not video_folder:
        print_error("Video download failed. Aborting pipeline.")
        sys.exit(1)
    
    # Step 3: Run operations pipeline with adaptive frame extraction
    headshot_results = run_operations_pipeline_with_adaptive_frames(video_folder, trained_actors)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    elapsed_minutes = elapsed_time / 60
    
    # Final summary
    print_header(f"\n=== FINAL RESULTS ===")
    print(f"Video folder: {video_folder}")
    print(f"Total execution time: {elapsed_minutes:.1f} minutes ({elapsed_time:.1f} seconds)")
    
    total_headshots = 0
    for actor_name in trained_actors:
        headshot_count = headshot_results.get(actor_name, 0)
        total_headshots += headshot_count
        if headshot_count > 0:
            print(f"✓ {actor_name}: {headshot_count} headshots")
        else:
            print_error(f"✗ {actor_name}: No headshots found")
    
    if failed_actors:
        print_error(f"Training failed: {', '.join(failed_actors)}")
    
    if total_headshots > 0:
        print(f"🎉 SUCCESS! Found {total_headshots} total headshots across all actors")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
