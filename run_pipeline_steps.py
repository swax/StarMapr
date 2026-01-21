#!/usr/bin/env python3
"""
StarMapr Pipeline Runner

Interactive script to run the complete actor face recognition pipeline.
Provides a numbered menu of pipeline steps for a given actor.
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from utils import get_average_embedding_path, get_venv_python

# Load environment variables
load_dotenv()

# Pipeline configuration
TOTAL_PIPELINE_STEPS = 15


def get_actor_name():
    """
    Get actor name from user input.
    
    Returns:
        str: Actor name entered by user
    """
    while True:
        name = input("\nEnter actor name (e.g., 'Bill Murray'): ").strip()
        if name:
            return name
        print("Please enter a valid actor name.")


def get_show_name():
    """
    Get optional show name from user input.
    
    Returns:
        str or None: Show name entered by user, or None if skipped
    """
    show = input("Enter show/movie name (e.g., 'SNL'): ").strip()
    return show if show else None


def get_page_number():
    """
    Get page number for image download.
    
    Returns:
        int: Page number (defaults to 1 if no input)
    """
    while True:
        try:
            page_input = input("Page number to download (press Enter for page 1): ").strip()
            if not page_input:
                return 1  # Default to page 1
            page = int(page_input)
            if page > 0:
                return page
            print("Please enter a positive page number.")
        except ValueError:
            print("Please enter a valid page number.")


def display_menu(actor_name):
    """
    Display the pipeline menu options.
    
    Args:
        actor_name (str): Name of the actor
    """
    print(f"\n=== StarMapr Pipeline for '{actor_name}' ===")
    print("Select a step to run:")
    print()
    print("TRAINING PIPELINE:")
    print("1. Download training images (solo portraits)")
    print("2. Remove duplicate training images")
    print("3. Remove bad training images (keep exactly 1 face)")
    print("4. Remove outlier faces (detect inconsistent faces)")
    print("5. Compute average embeddings")
    print()
    print("TESTING PIPELINE:")
    print("6. Download testing images (group photos)")
    print("7. Remove duplicate testing images")
    print("8. Remove bad testing images (keep 4-10 faces)")
    print("9. Detect faces in test images")
    print("10. Accept model (copy to models directory)")
    print()
    print("OPERATIONS PIPELINE:")
    print("11. Download video from URL")
    print("12. Extract frames from video")
    print("13. Extract faces from video frames")
    print("14. Extract actor headshots from video")
    print()
    print(f"{TOTAL_PIPELINE_STEPS}. Exit")
    print()


def copy_model_to_models_dir(actor_name):
    """
    Copy the average embedding file from training directory to models directory.
    
    Args:
        actor_name (str): Name of the actor
        
    Returns:
        bool: True if successful, False if failed
    """
    try:
        import shutil
        
        # Get paths using utility functions
        source_path = get_average_embedding_path(actor_name, 'training')
        dest_path = get_average_embedding_path(actor_name, 'models')
        
        # Check if source exists
        if not source_path.exists():
            print(f"✗ Source embedding file not found: {source_path}")
            print("Make sure you have run 'Compute average embeddings' first.")
            return False
        
        # Create models directory if it doesn't exist
        dest_path.parent.mkdir(exist_ok=True)
        
        # Copy file to models directory
        shutil.copy2(source_path, dest_path)
        
        print(f"✓ Model successfully copied to: {dest_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to copy model file: {e}")
        return False



def run_command(command, description):
    """
    Run a shell command and display the result.
    
    Args:
        command (list): Command to run as list of arguments
        description (str): Description of what the command does
        
    Returns:
        tuple: (success: bool, last_line: str or None)
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    try:
        # Create fresh environment with reloaded .env variables
        env = os.environ.copy()
        # Re-read .env file to get any changes made while script is running
        from dotenv import dotenv_values
        env_vars = dotenv_values('.env')
        env.update(env_vars)
        
        # Always capture output to get the last line
        result = subprocess.run(command, check=True, capture_output=True, text=True, env=env, encoding='utf-8', errors='replace')
        
        # Get the last non-empty line of output
        stdout_lines = result.stdout.strip().split('\n')
        last_line = stdout_lines[-1] if stdout_lines else ""
        
        print(f"\n✓ {description} completed successfully!")
        return True, last_line
            
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False, None
    except FileNotFoundError:
        print(f"\n✗ Command not found: {command[0]}")
        print("Make sure all required Python scripts are in the current directory.")
        return False, None


def main():
    """
    Main interactive pipeline runner.
    """
    print("Welcome to StarMapr Pipeline Runner!")
    
    # Get actor name and optional show
    actor_name = get_actor_name()
    show_name = get_show_name()
    
    # Track page numbers for download steps
    # Track video folder path for video processing steps
    video_folder_path = None
    # Track last step number for sequential execution
    last_step = None
    
    while True:
        display_menu(actor_name)
        
        try:
            if last_step and last_step < TOTAL_PIPELINE_STEPS:
                next_step = last_step + 1
                prompt = f"Enter your choice (1-{TOTAL_PIPELINE_STEPS}, or press Enter for step {next_step}): "
            else:
                prompt = f"Enter your choice (1-{TOTAL_PIPELINE_STEPS}): "
            
            choice = input(prompt).strip()
            
            # If no choice entered and we have a next step, use it
            if not choice and last_step and last_step < TOTAL_PIPELINE_STEPS:
                choice = str(last_step + 1)
            
            if choice == '1':
                # Download training images
                training_page = get_page_number()
                command = [get_venv_python(), 'download_actor_images.py', actor_name, '--training', '--show', show_name]
                if training_page != 1:
                    command.extend(['--page', str(training_page)])
                success, _ = run_command(command, "Download training images")
                
            elif choice == '2':
                # Remove duplicate training images
                command = [get_venv_python(), 'remove_dupe_training_images.py', '--training', actor_name]
                success, _ = run_command(command, "Remove duplicate training images")
                
            elif choice == '3':
                # Remove bad training images
                command = [get_venv_python(), 'remove_bad_training_images.py', '--training', actor_name]
                success, _ = run_command(command, "Remove bad training images")
                
            elif choice == '4':
                # Remove outlier faces
                command = [get_venv_python(), 'remove_face_outliers.py', '--training', actor_name]
                success, _ = run_command(command, "Remove outlier faces")
                
            elif choice == '5':
                # Compute average embeddings
                command = [get_venv_python(), 'compute_average_embeddings.py', actor_name]
                success, _ = run_command(command, "Compute average embeddings")
                
            elif choice == '6':
                # Download testing images
                testing_page = get_page_number()
                command = [get_venv_python(), 'download_actor_images.py', actor_name, '--testing', '--show', show_name]
                if testing_page != 1:
                    command.extend(['--page', str(testing_page)])
                success, _ = run_command(command, "Download testing images")
                
            elif choice == '7':
                # Remove duplicate testing images
                command = [get_venv_python(), 'remove_dupe_training_images.py', '--testing', actor_name]
                success, _ = run_command(command, "Remove duplicate testing images")
                
            elif choice == '8':
                # Remove bad testing images
                command = [get_venv_python(), 'remove_bad_training_images.py', '--testing', actor_name]
                success, _ = run_command(command, "Remove bad testing images")
                
            elif choice == '9':
                # Detect faces
                command = [get_venv_python(), 'eval_star_detection.py', actor_name]
                success, _ = run_command(command, "Detect faces in test images")
                
            elif choice == '10':
                # Accept model (copy to models directory)
                print(f"\n{'='*60}")
                print("Accepting model: Copy average embedding to models directory")
                print(f"{'='*60}")
                if copy_model_to_models_dir(actor_name):
                    print(f"\n✓ Model acceptance completed successfully!")
                    last_step = 10
                else:
                    print(f"\n✗ Model acceptance failed!")
                
            elif choice == '11':
                # Download video from URL
                video_url = input("\nEnter video URL (YouTube, Vimeo, TikTok, etc.): ").strip()
                if not video_url:
                    print("No URL provided.")
                    continue
                command = [get_venv_python(), 'download_video.py', video_url]
                success, last_line = run_command(command, "Download video from URL")
                # Parse JSON output to get video folder path
                if success and last_line:
                    try:
                        import json
                        json_result = json.loads(last_line)
                        if json_result.get("success"):
                            video_folder_path = json_result.get("video_folder")
                            print(f"Video folder: {video_folder_path}")
                    except json.JSONDecodeError:
                        pass  # Not JSON, ignore
                
            elif choice == '12':
                # Extract frames from video
                if video_folder_path:
                    default_prompt = f"\nEnter path to video folder (default: {video_folder_path}): "
                else:
                    default_prompt = "\nEnter path to video folder (e.g., 05_videos/youtube_ABC123/): "
                
                video_folder = input(default_prompt).strip()
                if not video_folder and video_folder_path:
                    video_folder = video_folder_path
                
                # Remember this video folder path for subsequent steps
                if video_folder:
                    video_folder_path = video_folder
                    
                # Get default frame count from environment variable
                default_frame_count = os.getenv('OPERATIONS_EXTRACT_FRAME_COUNT', '50')
                frame_count = input(f"Number of frames to extract (default {default_frame_count}): ").strip()
                if not frame_count:
                    # Use environment default by not passing frame count (let script handle it)
                    command = [get_venv_python(), 'extract_video_frames.py', video_folder]
                else:
                    command = [get_venv_python(), 'extract_video_frames.py', video_folder, frame_count]
                success, _ = run_command(command, "Extract frames from video")
                
            elif choice == '13':
                # Extract faces from video frames
                if video_folder_path:
                    default_prompt = f"\nEnter path to video folder (default: {video_folder_path}): "
                else:
                    default_prompt = "\nEnter path to video folder (e.g., 05_videos/youtube_ABC123/): "
                
                video_folder = input(default_prompt).strip()
                if not video_folder and video_folder_path:
                    video_folder = video_folder_path
                
                command = [get_venv_python(), 'extract_frame_faces.py', video_folder]
                success, _ = run_command(command, "Extract faces from video frames")
                
            elif choice == '14':
                # Extract actor headshots from video
                if video_folder_path:
                    default_prompt = f"\nEnter path to video folder (default: {video_folder_path}): "
                else:
                    default_prompt = f"\nEnter path to video folder (e.g., 05_videos/youtube_ABC123/): "
                
                video_folder = input(default_prompt).strip()
                if not video_folder and video_folder_path:
                    video_folder = video_folder_path
                
                command = [get_venv_python(), 'extract_video_headshots.py', actor_name, video_folder]
                success, _ = run_command(command, f"Extract {actor_name} headshots from video")
                
            elif choice == str(TOTAL_PIPELINE_STEPS):
                print("\nExiting StarMapr Pipeline Runner. Goodbye!")
                sys.exit(0)
                
            else:
                print(f"\nInvalid choice: {choice}. Please enter a number between 1 and {TOTAL_PIPELINE_STEPS}.")
            
            # Update last_step for successful runs (except exit)
            if choice != str(TOTAL_PIPELINE_STEPS) and choice != '10' and 'success' in locals() and success:
                last_step = int(choice)
                
        except KeyboardInterrupt:
            print("\n\nExiting StarMapr Pipeline Runner. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()
