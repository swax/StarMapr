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
        int: Number of images to download
    """
    default_count = 15 if mode == 'training' else 30
    while True:
        try:
            count_input = input(f"Number of images to download for {mode} (default {default_count}): ").strip()
            if not count_input:
                return default_count
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
    print("10. Exit")
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
    
    while True:
        display_menu(celebrity_name)
        
        try:
            choice = input("Enter your choice (1-10): ").strip()
            
            if choice == '1':
                # Download training images
                if training_count is None:
                    training_count = get_image_count('training')
                command = ['python3', 'download_celebrity_images.py', celebrity_name, str(training_count), '--training']
                run_command(command, "Download training images")
                
            elif choice == '2':
                # Remove duplicate training images
                command = ['python3', 'remove_dupe_training_images.py', '--training', celebrity_name]
                run_command(command, "Remove duplicate training images")
                
            elif choice == '3':
                # Remove bad training images
                command = ['python3', 'remove_bad_training_images.py', '--training', celebrity_name]
                run_command(command, "Remove bad training images")
                
            elif choice == '4':
                # Remove outlier faces
                command = ['python3', 'remove_face_outliers.py', '--training', celebrity_name]
                run_command(command, "Remove outlier faces")
                
            elif choice == '5':
                # Compute average embeddings
                training_path = f'training/{celebrity_folder}/'
                if not Path(training_path).exists():
                    print(f"\n✗ Training folder not found: {training_path}")
                    print("Please run steps 1-3 first to create and clean training data.")
                    continue
                command = ['python3', 'compute_average_embeddings.py', training_path]
                run_command(command, "Compute average embeddings")
                
            elif choice == '6':
                # Download testing images
                if testing_count is None:
                    testing_count = get_image_count('testing')
                command = ['python3', 'download_celebrity_images.py', celebrity_name, str(testing_count), '--testing']
                run_command(command, "Download testing images")
                
            elif choice == '7':
                # Remove duplicate testing images
                command = ['python3', 'remove_dupe_training_images.py', '--testing', celebrity_name]
                run_command(command, "Remove duplicate testing images")
                
            elif choice == '8':
                # Remove bad testing images
                command = ['python3', 'remove_bad_training_images.py', '--testing', celebrity_name]
                run_command(command, "Remove bad testing images")
                
            elif choice == '9':
                # Detect faces
                testing_path = f'testing/{celebrity_folder}/'
                embedding_path = f'training/{celebrity_folder}/{celebrity_folder}_average_embedding.pkl'
                
                if not Path(testing_path).exists():
                    print(f"\n✗ Testing folder not found: {testing_path}")
                    print("Please run steps 5-7 first to create and clean testing data.")
                    continue
                    
                if not Path(embedding_path).exists():
                    print(f"\n✗ Embedding file not found: {embedding_path}")
                    print("Please run step 4 first to generate average embeddings.")
                    continue
                
                command = ['python3', 'eval_star_detection.py', testing_path, embedding_path]
                run_command(command, "Detect faces in test images")
                
            elif choice == '10':
                print("\nExiting StarMapr Pipeline Runner. Goodbye!")
                sys.exit(0)
                
            else:
                print(f"\nInvalid choice: {choice}. Please enter a number between 1 and 10.")
                
        except KeyboardInterrupt:
            print("\n\nExiting StarMapr Pipeline Runner. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()