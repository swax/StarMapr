#!/usr/bin/env python3
"""
Celebrity Image Downloader for StarMapr

Downloads celebrity images from Google Image Search to build training datasets.
Integrates with the existing training/ folder structure.
"""

import os
import argparse
import sys
import uuid
from google_images_search import GoogleImagesSearch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def download_celebrity_images(celebrity_name, num_images=10, mode='training', api_key=None, search_engine_id=None):
    """
    Download celebrity images from Google Image Search
    
    Args:
        celebrity_name (str): Name of the celebrity to search for
        num_images (int): Number of images to download
        mode (str): 'training' for solo portraits or 'testing' for group photos
        api_key (str): Google Custom Search API key
        search_engine_id (str): Google Custom Search Engine ID
    """
    
    # Get API credentials from environment or parameters
    if not api_key:
        api_key = os.getenv('GOOGLE_API_KEY')
    if not search_engine_id:
        search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
    
    if not api_key or not search_engine_id:
        print("Error: Missing API credentials. Please set your keys in the .env file:")
        print("GOOGLE_API_KEY=your_api_key_here")
        print("GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here")
        return False
    
    # Set up Google Images Search
    try:
        gis = GoogleImagesSearch(api_key, search_engine_id)
    except Exception as e:
        print(f"Error initializing Google Images Search: {e}")
        return False
    
    # Create celebrity directory based on mode
    celebrity_folder = celebrity_name.lower().replace(' ', '_')
    download_path = f'./{mode}/{celebrity_folder}/'
    os.makedirs(download_path, exist_ok=True)
    
    # Search parameters based on mode
    if mode == 'training':
        # Optimized for solo celebrity portraits
        search_params = {
            'q': f'{celebrity_name} face portrait',
            'num': num_images,
            'fileType': 'jpg|jpeg|png',
            'safe': 'medium',
            'imgType': 'face',
            'imgSize': 'medium'
        }
    else:  # testing mode
        # Optimized for group photos containing the celebrity
        search_params = {
            'q': f'{celebrity_name} group photo event',
            'num': num_images,
            'fileType': 'jpg|jpeg|png',
            'safe': 'medium',
            'imgType': 'photo',
            'imgSize': 'medium'
        }
    
    try:
        # Perform search and download
        print(f"Searching for {num_images} images of '{celebrity_name}'...")
        gis.search(search_params=search_params, path_to_dir=download_path)
        
        # Get downloaded files and rename them with GUID prefixes
        downloaded_files = [f for f in os.listdir(download_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Rename files using first 8 characters of GUID
        renamed_count = 0
        for filename in sorted(downloaded_files):
            old_path = os.path.join(download_path, filename)
            # Get file extension
            ext = os.path.splitext(filename)[1].lower()
            # Create new GUID-based filename (first 8 characters)
            guid_prefix = str(uuid.uuid4()).replace('-', '')[:8]
            new_filename = f"{guid_prefix}{ext}"
            new_path = os.path.join(download_path, new_filename)
            
            try:
                os.rename(old_path, new_path)
                renamed_count += 1
                print(f"  Renamed: {filename} → {new_filename}")
            except OSError as e:
                print(f"  Warning: Could not rename {filename}: {e}")
        
        print(f"Successfully downloaded {len(downloaded_files)} images for '{celebrity_name}'")
        print(f"Renamed {renamed_count} files with GUID-based names")
        print(f"Images saved to: {download_path}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading images: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download celebrity images from Google Image Search for StarMapr'
    )
    
    parser.add_argument('celebrity_name', 
                       help='Name of the celebrity (e.g., "Bill Murray")')
    
    parser.add_argument('num_images', type=int, 
                       help='Number of images to download')
    
    # Mutually exclusive group for mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--training', action='store_true',
                           help='Download solo portraits for training dataset')
    mode_group.add_argument('--testing', action='store_true', 
                           help='Download group photos for testing dataset')
    
    parser.add_argument('--api-key', 
                       help='Google Custom Search API key (or set GOOGLE_API_KEY env var)')
    
    parser.add_argument('--search-engine-id', 
                       help='Google Custom Search Engine ID (or set GOOGLE_SEARCH_ENGINE_ID env var)')
    
    args = parser.parse_args()
    
    # Validate input
    if args.num_images <= 0:
        print("Error: Number of images must be positive")
        sys.exit(1)
    
    if args.num_images > 100:
        print("Warning: Large number of images requested. Google API has daily limits.")
    
    # Determine mode from arguments
    mode = 'training' if args.training else 'testing'
    
    # Download images
    success = download_celebrity_images(
        celebrity_name=args.celebrity_name,
        num_images=args.num_images,
        mode=mode,
        api_key=args.api_key,
        search_engine_id=args.search_engine_id
    )
    
    if success:
        celebrity_folder = args.celebrity_name.lower().replace(' ', '_')
        print("\nNext steps:")
        if mode == 'training':
            print(f"1. Review downloaded images in training/{celebrity_folder}/")
            print("2. Remove any irrelevant or low-quality images")
            print(f"3. Run: python compute_average_embeddings.py training/{celebrity_folder}/")
        else:  # testing mode
            print(f"1. Review downloaded images in testing/{celebrity_folder}/")
            print("2. Use these group photos to test face detection accuracy")
            print(f"3. Run: python detect_star.py testing/{celebrity_folder}/ training/{celebrity_folder}/{celebrity_folder}_average_embedding.pkl")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()