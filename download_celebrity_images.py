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
from utils import get_celebrity_folder_path, get_env_int, ensure_folder_exists, print_error, print_summary

# Load environment variables from .env file
load_dotenv()


def download_celebrity_images(celebrity_name, num_images, mode='training', show=None, api_key=None, search_engine_id=None):
    """
    Download celebrity images from Google Image Search
    
    Args:
        celebrity_name (str): Name of the celebrity to search for
        num_images (int): Number of images to download
        mode (str): 'training' for solo portraits or 'testing' for group photos
        show (str): Optional show name to include in search. If provided, splits downloads between normal and show-specific queries
        api_key (str): Google Custom Search API key
        search_engine_id (str): Google Custom Search Engine ID
    """
    
    # Get API credentials from environment or parameters
    if not api_key:
        api_key = os.getenv('GOOGLE_API_KEY')
    if not search_engine_id:
        search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
    
    
    if not api_key or not search_engine_id:
        print_error("Missing API credentials. Please set your keys in the .env file:")
        print("GOOGLE_API_KEY=your_api_key_here")
        print("GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here")
        return False
    
    # Set up Google Images Search
    try:
        gis = GoogleImagesSearch(api_key, search_engine_id)
    except Exception as e:
        print_error(f"Error initializing Google Images Search: {e}")
        return False
    
    # Create celebrity directory based on mode
    download_path = get_celebrity_folder_path(celebrity_name, mode)
    ensure_folder_exists(download_path)
    
    def create_search_params(query, num_imgs):
        """Create search parameters based on mode"""
        if mode == 'training':
            return {
                'q': query,
                'num': num_imgs,
                'fileType': 'jpg|jpeg|png',
                'safe': 'medium',
                'imgType': 'face',
                'imgSize': 'medium'
            }
        else:  # testing mode
            return {
                'q': query,
                'num': num_imgs,
                'fileType': 'jpg|jpeg|png',
                'safe': 'medium',
                'imgType': 'photo',
                'imgSize': 'large'
            }
    
    query_suffix = 'face' if mode == 'training' else 'group'

    # Determine search queries and image counts
    if show:
        # Split downloads between normal and show-specific queries
        normal_images = num_images // 2
        show_images = num_images - normal_images
        
        searches = []
        searches.append((f'{celebrity_name} {query_suffix}', normal_images, 'normal'))
        searches.append((f'{celebrity_name} {show} {query_suffix}', show_images, f'show-specific ({show})'))
        
        print(f"Downloading {normal_images} normal and {show_images} show-specific images for '{celebrity_name}'...")
    else:
        # Single search without show parameter
        searches = [(f'{celebrity_name} {query_suffix}', num_images, 'normal')]
        print(f"Searching for {num_images} images of '{celebrity_name}'...")
    
    try:
        # Perform searches and downloads
        for query, num_imgs, search_type in searches:
            if len(searches) > 1:
                print(f"  Downloading {num_imgs} {search_type} images...")
            search_params = create_search_params(query, num_imgs)
            gis.search(search_params=search_params, path_to_dir=download_path)
        
        # Get downloaded files and rename them with GUID prefixes
        downloaded_files = [f for f in os.listdir(download_path)]
        
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
                print_error(f"Warning: Could not rename {filename}: {e}")
        
        print(f"Renamed {renamed_count} files with GUID-based names")

        print_summary(f"Successfully downloaded {len(downloaded_files)} images for '{celebrity_name}' - Images saved to: {download_path}")
        
        return True
        
    except Exception as e:
        print_error(f"Error downloading images: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download celebrity images from Google Image Search for StarMapr'
    )
    
    parser.add_argument('celebrity_name', 
                       help='Name of the celebrity (e.g., "Bill Murray")')
    
    parser.add_argument('num_images', type=int, nargs='?',
                       help='Number of images to download (default: from TRAINING_IMAGE_COUNT/TESTING_IMAGE_COUNT env vars)')
    
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
    
    parser.add_argument('--show', 
                       help='Optional show name to include in search. Downloads half normal images, half show-specific')
    
    args = parser.parse_args()
    
    # Get default image count if not provided
    num_images = args.num_images
    if num_images is None:
        mode = 'training' if args.training else 'testing'
        if mode == 'training':
            num_images = get_env_int('TRAINING_IMAGE_COUNT', 20)
        else:
            num_images = get_env_int('TESTING_IMAGE_COUNT', 30)
    
    # Validate input
    if num_images <= 0:
        print_error("Number of images must be positive")
        sys.exit(1)
    
    if num_images > 100:
        print("Warning: Large number of images requested. Google API has daily limits.")
    
    # Determine mode from arguments
    mode = 'training' if args.training else 'testing'
    
    # Download images
    success = download_celebrity_images(
        celebrity_name=args.celebrity_name,
        num_images=num_images,
        mode=mode,
        show=args.show,
        api_key=args.api_key,
        search_engine_id=args.search_engine_id
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()