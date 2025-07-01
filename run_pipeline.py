#!/usr/bin/env python3
"""
StarMapr Pipeline Runner

Interactive script to run the complete celebrity face recognition pipeline.
Provides a numbered menu of pipeline steps for a given celebrity.
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pipeline configuration
TOTAL_PIPELINE_STEPS = 14


def get_celebrity_name():
    """
    Get celebrity name from user input.
    
    Returns:
        str: Celebrity name entered by user
    """
    while True:
        name = input("\nEnter celebrity name (e.g., 'Bill Murray'): ").strip()
        if name:
            return name
        print("Please enter a valid celebrity name.")


def get_image_count(mode):
    """
    Get number of images to download for the given mode.
    
    Args:
        mode (str): 'training' or 'testing'
        
    Returns:
        int or None: Number of images to download, or None to use .env default
    """
    if mode == 'training':
        env_default = int(os.getenv('TRAINING_IMAGE_COUNT', 20))
    else:
        env_default = int(os.getenv('TESTING_IMAGE_COUNT', 30))
    
    while True:
        try:
            count_input = input(f"Number of images to download for {mode} (press Enter for default {env_default}): ").strip()
            if not count_input:
                return None  # Use .env default
            count = int(count_input)
            if count > 0:
                return count
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")


def display_menu(celebrity_name):
    """
    Display the pipeline menu options.
    
    Args:
        celebrity_name (str): Name of the celebrity
    """
    celebrity_folder = celebrity_name.lower().replace(' ', '_')
    
    print(f"\n=== StarMapr Pipeline for '{celebrity_name}' ===")
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
    print()
    print("VIDEO PROCESSING PIPELINE:")
    print("10. Download video from URL")
    print("11. Extract frames from video")
    print("12. Extract faces from video frames")
    print("13. Extract celebrity headshots from video")
    print()
    print(f"{TOTAL_PIPELINE_STEPS}. Exit")
    print()


def run_command(command, description):
    """
    Run a shell command and display the result.
    
    Args:
        command (list): Command to run as list of arguments
        description (str): Description of what the command does
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=False)
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Command not found: {command[0]}")
        print("Make sure all required Python scripts are in the current directory.")
        return False


def main():
    """
    Main interactive pipeline runner.
    """
    print("Welcome to StarMapr Pipeline Runner!")
    
    # Get celebrity name
    celebrity_name = get_celebrity_name()
    celebrity_folder = celebrity_name.lower().replace(' ', '_')
    
    # Track image counts for download steps
    training_count = None
    testing_count = None
    # Track video folder path for video processing steps
    video_folder_path = None
    # Track last step number for sequential execution
    last_step = None
    
    while True:
        display_menu(celebrity_name)
        
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
                if training_count is None:
                    training_count = get_image_count('training')
                if training_count is None:
                    command = ['python3', 'download_celebrity_images.py', celebrity_name, '--training']
                else:
                    command = ['python3', 'download_celebrity_images.py', celebrity_name, str(training_count), '--training']
                if run_command(command, "Download training images"):
                    last_step = 1
                
            elif choice == '2':
                # Remove duplicate training images
                command = ['python3', 'remove_dupe_training_images.py', '--training', celebrity_name]
                if run_command(command, "Remove duplicate training images"):
                    last_step = 2
                
            elif choice == '3':
                # Remove bad training images
                command = ['python3', 'remove_bad_training_images.py', '--training', celebrity_name]
                if run_command(command, "Remove bad training images"):
                    last_step = 3
                
            elif choice == '4':
                # Remove outlier faces
                command = ['python3', 'remove_face_outliers.py', '--training', celebrity_name]
                if run_command(command, "Remove outlier faces"):
                    last_step = 4
                
            elif choice == '5':
                # Compute average embeddings
                command = ['python3', 'compute_average_embeddings.py', celebrity_name]
                if run_command(command, "Compute average embeddings"):
                    last_step = 5
                
            elif choice == '6':
                # Download testing images
                if testing_count is None:
                    testing_count = get_image_count('testing')
                if testing_count is None:
                    command = ['python3', 'download_celebrity_images.py', celebrity_name, '--testing']
                else:
                    command = ['python3', 'download_celebrity_images.py', celebrity_name, str(testing_count), '--testing']
                if run_command(command, "Download testing images"):
                    last_step = 6
                
            elif choice == '7':
                # Remove duplicate testing images
                command = ['python3', 'remove_dupe_training_images.py', '--testing', celebrity_name]
                if run_command(command, "Remove duplicate testing images"):
                    last_step = 7
                
            elif choice == '8':
                # Remove bad testing images
                command = ['python3', 'remove_bad_training_images.py', '--testing', celebrity_name]
                if run_command(command, "Remove bad testing images"):
                    last_step = 8
                
            elif choice == '9':
                # Detect faces
                command = ['python3', 'eval_star_detection.py', celebrity_name]
                if run_command(command, "Detect faces in test images"):
                    last_step = 9
                
            elif choice == '10':
                # Download video from URL
                video_url = input("\nEnter video URL (YouTube, Vimeo, TikTok, etc.): ").strip()
                if not video_url:
                    print("No URL provided.")
                    continue
                command = ['python3', 'download_video.py', video_url]
                if run_command(command, "Download video from URL"):
                    last_step = 10
                
            elif choice == '11':
                # Extract frames from video
                if video_folder_path:
                    default_prompt = f"\nEnter path to video folder (default: {video_folder_path}): "
                else:
                    default_prompt = "\nEnter path to video folder (e.g., videos/youtube_ABC123/): "
                
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
                    command = ['python3', 'extract_video_frames.py', video_folder]
                else:
                    command = ['python3', 'extract_video_frames.py', video_folder, frame_count]
                if run_command(command, "Extract frames from video"):
                    last_step = 11
                
            elif choice == '12':
                # Extract faces from video frames
                if video_folder_path:
                    default_prompt = f"\nEnter path to video folder (default: {video_folder_path}): "
                else:
                    default_prompt = "\nEnter path to video folder (e.g., videos/youtube_ABC123/): "
                
                video_folder = input(default_prompt).strip()
                if not video_folder and video_folder_path:
                    video_folder = video_folder_path
                
                command = ['python3', 'extract_frame_faces.py', video_folder]
                if run_command(command, "Extract faces from video frames"):
                    last_step = 12
                
            elif choice == '13':
                # Extract celebrity headshots from video
                if video_folder_path:
                    default_prompt = f"\nEnter path to video folder (default: {video_folder_path}): "
                else:
                    default_prompt = f"\nEnter path to video folder (e.g., videos/youtube_ABC123/): "
                
                video_folder = input(default_prompt).strip()
                if not video_folder and video_folder_path:
                    video_folder = video_folder_path
                
                command = ['python3', 'extract_video_headshots.py', celebrity_name, video_folder]
                if run_command(command, f"Extract {celebrity_name} headshots from video"):
                    last_step = 13
                
            elif choice == str(TOTAL_PIPELINE_STEPS):
                print("\nExiting StarMapr Pipeline Runner. Goodbye!")
                sys.exit(0)
                
            else:
                print(f"\nInvalid choice: {choice}. Please enter a number between 1 and {TOTAL_PIPELINE_STEPS}.")
                
        except KeyboardInterrupt:
            print("\n\nExiting StarMapr Pipeline Runner. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()