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


def download_celebrity_images(celebrity_name, mode='training', show=None, page=1, api_key=None, search_engine_id=None):
    """
    Download celebrity images from Google Image Search
    
    Downloads 20 images using different search queries for each page.
    Each page uses a different search term to get varied results.
    
    Args:
        celebrity_name (str): Name of the celebrity to search for
        mode (str): 'training' for solo portraits or 'testing' for group photos
        show (str): Show name to include in search for targeted results
        page (int): Page number (1-5) - each uses different search terms
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
    
    #*** Actually paging Google Search results doesn't work at all, the start parameter is ignored.

    # Define search terms for different pages
    if mode == 'training':
        search_terms = [
            f'{celebrity_name} {show}',    # Page 1: name + show
            celebrity_name,                # Page 2: just the name
            f'{celebrity_name} face',      # Page 3: name + face
            f'{celebrity_name} headshot',  # Page 4: name + headshot
            f'{celebrity_name} portrait'   # Page 5: name + portrait
        ]
    else:  # testing mode
        search_terms = [
            f'{celebrity_name} group',         # Page 1: name + group
            f'{celebrity_name} cast',          # Page 2: name + cast
            f'{celebrity_name} team',          # Page 3: name + team
            f'{celebrity_name} with friends',  # Page 4: name + with friends
            f'{celebrity_name} ensemble'       # Page 5: name + ensemble
        ]
    
    # Get search query for this page
    if page > len(search_terms):
        print_error(f"Page {page} is not supported. Maximum page is {len(search_terms)}.")
        return False
    
    query = search_terms[page - 1]

    # Needs to be increments of 10, google queries in blocks of 10
    images_to_download = 20
    
    print(f"Downloading page {page} images for '{celebrity_name}' using query: '{query}' (20 total)...")
    
    # Get initial file count to track new downloads
    initial_files = set(os.listdir(download_path))
    
    try:
        # Perform searches and downloads
        search_params = {
            'q': query,
            'num': images_to_download,
            'fileType': 'jpg|jpeg|png',
            'imgSize': 'medium' if mode == 'training' else 'large',
        }
            
        gis.search(search_params=search_params, path_to_dir=download_path)
    
        # Get all files after download and identify newly downloaded ones
        all_files = set(os.listdir(download_path))
        newly_downloaded_files = list(all_files - initial_files)
        print(f"Actually downloaded {len(newly_downloaded_files)} new files")
        
        # Rename only newly downloaded files using first 8 characters of GUID
        renamed_count = 0
        for filename in sorted(newly_downloaded_files):
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

        print_summary(f"Successfully downloaded {len(newly_downloaded_files)} new images for '{celebrity_name}' - Images saved to: {download_path}")
        
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
    
    parser.add_argument('--show', required=True,
                       help='Show name to include in search. Downloads 10 general + 10 show-specific images (20 total)')
    
    parser.add_argument('--page', type=int, default=1,
                       help='Page number (1-5, default: 1). Each page uses different search terms to download 20 images.')
    
    args = parser.parse_args()
    
    # Validate input
    if args.page <= 0:
        print_error("Page number must be positive")
        sys.exit(1)
    
    if args.page > 5:
        print_error("Page number must be 1-5. Each page uses different search terms.")
        sys.exit(1)
    
    # Determine mode from arguments
    mode = 'training' if args.training else 'testing'
    
    # Download images
    success = download_celebrity_images(
        celebrity_name=args.celebrity_name,
        mode=mode,
        show=args.show,
        page=args.page,
        api_key=args.api_key,
        search_engine_id=args.search_engine_id
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()