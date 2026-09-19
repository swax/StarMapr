#!/usr/bin/env python3
"""
StarMapr Headshot Detection Script

Automated script that takes a video URL and a list of actors, runs actor 
training for each, then downloads the video and extracts headshots for all successfully 
trained actors. Uses adaptive frame extraction if no headshots are initially found.

Usage:
    python3 01_run_headshot_detection.py "https://youtube.com/watch?v=VIDEO_ID" --show "SNL" "Bill Murray" "Tina Fey" "Amy Poehler"
    python3 01_run_headshot_detection.py "https://youtube.com/watch?v=VIDEO_ID" --show "SNL" --actors "Bill Murray,Tina Fey,Amy Poehler"
"""

import os
import sys
import subprocess
import argparse
import json
import re
import time
from pathlib import Path
from progress import run_logged, emit_progress, initialize_progress
from readiness import check_readiness
from dotenv import load_dotenv
from utils import (
    log, get_actor_folder_name, get_env_int, print_error, get_venv_python
)
from validation import write_json

# Load environment variables
load_dotenv()


def emit_json_result(payload):
    """Emit a machine-readable result for API wrappers and automation."""
    print(json.dumps(payload, ensure_ascii=True))


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
        actor = actor.strip()
        if actor and actor not in seen:
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
        result = run_logged(command_list, check=True)
        
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
    
    command = [get_venv_python(), '02_run_actor_training.py', actor_name, show_name]
    
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
    
    command = [get_venv_python(), '30_download_video.py', video_url]
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
    
    command = [get_venv_python(), '31_extract_video_frames.py', video_folder, str(frame_count)]
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
    
    command = [get_venv_python(), '32_extract_frame_faces.py', video_folder]
    success, _, _ = run_subprocess_command(command, "Extracting faces from frames")
    
    return success


def extract_actor_headshots(actor_name, video_folder):
    """
    Extract headshots for a specific actor from video.
    
    Args:
        actor_name (str): Name of the actor
        video_folder (str): Path to video folder
        
    Returns:
        tuple: (success: bool, result report)
    """
    print(f"Extracting headshots for {actor_name}...")
    
    command = [get_venv_python(), '33_extract_video_headshots.py', actor_name, video_folder]
    success, _, _ = run_subprocess_command(command, f"Extracting {actor_name} headshots")
    
    if not success:
        return False, dict(status='extraction_error', retryable=False, headshots=[])
    report_path = Path(video_folder) / 'headshots' / get_actor_folder_name(actor_name) / 'result.json'
    try:
        report = json.loads(report_path.read_text(encoding='utf-8'))
        for headshot in report['headshots']:
            if not (report_path.parent / headshot['file']).is_file():
                raise ValueError('Result references a missing headshot')
        return True, report
    except (OSError, ValueError, KeyError):
        return False, dict(status='invalid_result', retryable=False, headshots=[])


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
    outcomes = {}
    pending = list(trained_actors)
    os.environ['STARMAPR_VIDEO_ACTORS'] = json.dumps(trained_actors)
    
    for multiplier in range(1, max_multiplier + 1):
        current_frame_count = default_frame_count * multiplier
        
        if multiplier > 1:
            print(f"\n--- Attempt {multiplier}: {current_frame_count} frames ---")
        else:
            print(f"\n--- Initial attempt: {current_frame_count} frames ---")
        
        # Extract frames
        if not extract_frames_from_video(video_folder, current_frame_count):
            raise RuntimeError('Frame extraction failed')
        
        # Extract faces from frames
        if not extract_faces_from_frames(video_folder):
            raise RuntimeError('Face extraction failed')
        
        # Extract headshots for each trained actor
        total_headshots_found = 0
        for actor_name in pending:
            os.environ['STARMAPR_ACTOR'] = actor_name
            emit_progress('headshots', processed=len(outcomes), total=len(trained_actors), attempt=multiplier)
            success, report = extract_actor_headshots(actor_name, video_folder)
            if success:
                outcomes[actor_name] = report
                headshot_count = len(report['headshots'])
                results[actor_name] = headshot_count
                total_headshots_found += headshot_count
                if headshot_count > 0:
                    print(f"✓ Found {headshot_count} headshots for {actor_name}")
                else:
                    print(f"No headshots found for {actor_name}")
            else:
                raise RuntimeError(f"Failed to extract headshots for {actor_name}: {report['status']}")
        pending = [actor for actor in pending if outcomes[actor]['retryable']]
        write_json(Path(video_folder) / 'headshot-results.json', dict(actors=outcomes, attempt=multiplier))
        if not pending:
            break
        
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
    
    for report in outcomes.values():
        if report['retryable']:
            report.update(retryable=False, stop_reason='frame_attempt_limit')
            write_json(Path(video_folder) / 'headshots' / get_actor_folder_name(report['actor']) / 'result.json', report)
    write_json(Path(video_folder) / 'headshot-results.json', dict(actors=outcomes, attempt=multiplier))
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
    parser.add_argument('--json', action='store_true',
                       help='Emit a final JSON result line for automation')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show all output from subprocess commands')
    
    args = parser.parse_args()

    if args.verbose:
        os.environ['STARMAPR_LOG_VERBOSE'] = 'true'

    def finish(success, actors, trained_actors=None, failed_actors=None,
               video_folder=None, headshot_results=None, error=None):
        trained_actors = trained_actors or []
        failed_actors = failed_actors or []
        headshot_results = headshot_results or {}
        elapsed_time = time.time() - start_time

        payload = {
            "success": success,
            "video_url": args.video_url,
            "show": args.show,
            "actors": actors,
            "trained_actors": trained_actors,
            "failed_actors": failed_actors,
            "video_folder": video_folder,
            "headshot_results": headshot_results,
            "total_headshots": sum(headshot_results.values()),
            "elapsed_seconds": round(elapsed_time, 3),
            "headshot_outcomes": {},
        }

        if success and video_folder:
            try:
                summary = json.loads((Path(video_folder) / 'headshot-results.json').read_text(encoding='utf-8'))
                payload['headshot_outcomes'] = {
                    actor: summary['actors'][actor] for actor in actors
                    if actor in summary['actors'] and actor not in failed_actors
                }
            except (OSError, ValueError, KeyError, TypeError):
                success = False
                payload['success'] = False
                error = 'Missing or invalid headshot validation summary'
        payload['outcome'] = ('headshots_available' if payload['total_headshots'] else 'no_reliable_headshots') if success else 'failed'

        if error:
            payload["error"] = error

        if args.json:
            emit_json_result(payload)

        sys.exit(0 if success else 1)
    
    # Parse actor names
    actors = parse_actors(args.actors, args.actor_list)
    
    if not actors:
        error = "No actors specified. Use either positional arguments or --actors flag."
        print_error(error)
        finish(False, actors, error=error)
    
    print_header(f"=== HEADSHOT DETECTION PIPELINE ===")
    print(f"Video URL: {args.video_url}")
    print(f"Actors: {', '.join(actors)}")
    print(f"Show: {args.show}")

    initialize_progress()
    try:
        emit_progress('preflight', 'started')
        print(json.dumps(check_readiness()))
    except (RuntimeError, ImportError, ValueError) as exc:
        finish(False, actors, error=str(exc))
    # Establish real downloader/host access before any expensive actor training.
    video_folder = download_video(args.video_url)
    if not video_folder:
        finish(False, actors, error='Video download failed during preflight; training was not started')
    
    # Step 1: Run actor training for each actor
    trained_actors = []
    failed_actors = []
    
    for actor_name in actors:
        os.environ['STARMAPR_ACTOR'] = actor_name
        emit_progress('training', processed=len(trained_actors) + len(failed_actors), total=len(actors))
        if run_actor_training(actor_name, args.show):
            trained_actors.append(actor_name)
        else:
            failed_actors.append(actor_name)
    
    if not trained_actors:
        error = "No actors were successfully trained. Aborting pipeline."
        print_error(error)
        finish(False, actors, trained_actors, failed_actors, error=error)
    
    print_header(f"\nTraining Results:")
    print(f"✓ Successfully trained: {', '.join(trained_actors)}")
    if failed_actors:
        print(f"✗ Failed to train: {', '.join(failed_actors)}")
    
    # Step 3: Run operations pipeline with adaptive frame extraction
    try:
        headshot_results = run_operations_pipeline_with_adaptive_frames(video_folder, trained_actors)
    except (RuntimeError, OSError, ValueError) as exc:
        print_error(str(exc))
        finish(False, actors, trained_actors, failed_actors, video_folder, error=str(exc))
    
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
        finish(True, actors, trained_actors, failed_actors, video_folder, headshot_results)
    else:
        print('Completed: no reliable headshot. Continue without an optional portrait; see headshot-results.json.')
        finish(True, actors, trained_actors, failed_actors, video_folder, headshot_results)


if __name__ == '__main__':
    main()
