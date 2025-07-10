#!/usr/bin/env python3
"""
Actor Image Downloader for StarMapr

Downloads actor images from Google Image Search to build training datasets.
Integrates with the existing training/ folder structure.
"""

import os
import argparse
import sys
import uuid
import shutil
from google_images_search import GoogleImagesSearch
from dotenv import load_dotenv
from utils import get_actor_folder_name, get_actor_folder_path, get_env_int, ensure_folder_exists, print_error, print_summary, log

# Load environment variables from .env file
load_dotenv()


def copy_images_from_cache_to_destination(cache_folder, destination_folder):
    """
    Copy images from cache folder to destination folder, renaming with new GUIDs.
    
    Args:
        cache_folder (str): Path to cache folder containing images
        destination_folder (str): Path to destination folder
        
    Returns:
        int: Number of files successfully copied
    """
    ensure_folder_exists(destination_folder)
    
    # Get all image files from cache
    cached_files = os.listdir(cache_folder)
    
    copied_count = 0
    for filename in cached_files:
        source_path = os.path.join(cache_folder, filename)
        # Get file extension
        ext = os.path.splitext(filename)[1].lower()
        # Create new GUID-based filename (first 8 characters)
        guid_prefix = str(uuid.uuid4()).replace('-', '')[:8]
        new_filename = f"{guid_prefix}{ext}"
        dest_path = os.path.join(destination_folder, new_filename)
        
        try:
            shutil.copy2(source_path, dest_path)
            copied_count += 1
            log(f"  Copied: {filename} → {new_filename}")
        except Exception as e:
            print_error(f"Warning: Could not copy {filename}: {e}")
    
    return copied_count


def download_actor_images(actor_name, mode='training', show=None, page=1, api_key=None, search_engine_id=None):
    """
    Download actor images from Google Image Search
    
    Downloads 20 images using different search queries for each page.
    Each page uses a different search term to get varied results.
    
    Args:
        actor_name (str): Name of the actor to search for
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
        log("GOOGLE_API_KEY=your_api_key_here")
        log("GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here")
        return False
    
    # Set up Google Images Search
    try:
        gis = GoogleImagesSearch(api_key, search_engine_id)
    except Exception as e:
        print_error(f"Error initializing Google Images Search: {e}")
        return False
    
    # Create actor directory based on mode
    download_path = get_actor_folder_path(actor_name, mode)
    ensure_folder_exists(download_path)
    
    #*** Actually paging Google Search results doesn't work at all, the start parameter is ignored.

    # Define search terms for different pages
    if mode == 'training':
        search_terms = [
            f'{actor_name} {show}',    # Page 1: name + show
            actor_name,                # Page 2: just the name
            f'{actor_name} face',      # Page 3: name + face
            f'{actor_name} headshot',  # Page 4: name + headshot
            f'{actor_name} portrait'   # Page 5: name + portrait
        ]
    else:  # testing mode
        search_terms = [
            f'{actor_name} group',         # Page 1: name + group
            f'{actor_name} cast',          # Page 2: name + cast
            f'{actor_name} team',          # Page 3: name + team
            f'{actor_name} with friends',  # Page 4: name + with friends
            f'{actor_name} ensemble'       # Page 5: name + ensemble
        ]
    
    # Get search query for this page
    if page > len(search_terms):
        print_error(f"Page {page} is not supported. Maximum page is {len(search_terms)}.")
        return False
    
    query = search_terms[page - 1]

    # Generate cache key and check for cached images
    actor_folder = get_actor_folder_name(actor_name)
    cache_key = query.lower().replace(' ', '_')
    cache_folder = f'01_search_cache/{actor_folder}/{mode}/{cache_key}/'

    # Check if cache folder exists and has images
    if os.path.exists(cache_folder):
        cached_files = os.listdir(cache_folder)
        
        if cached_files:
            log(f"Found {len(cached_files)} cached images for query: '{query}'")
            log(f"Copying cached images to: {download_path}")
            
            # Use common function to copy cached files to destination
            copied_count = copy_images_from_cache_to_destination(cache_folder, download_path)
            
            print_summary(f"Successfully copied {copied_count} cached images for '{actor_name}' - Images saved to: {download_path}")
            return True

    # If actor name starts with 'mock_' return an error here as it should not get to this point
    if actor_name.lower().startswith('mock_'):
        print_error("Mock actor data not found")
        return False

    # Needs to be increments of 10, google queries in blocks of 10
    images_to_download = 20
    
    log(f"Downloading page {page} images for '{actor_name}' using query: '{query}' (20 total)...")
    
    # Ensure cache folder exists and download directly to cache
    ensure_folder_exists(cache_folder)
    
    # Get initial file count in cache to track new downloads
    initial_cache_files = set(os.listdir(cache_folder))
    
    try:
        # Perform searches and downloads directly to cache folder
        search_params = {
            'q': query,
            'num': images_to_download,
            'fileType': 'jpg|jpeg|png',
            'imgSize': 'medium' if mode == 'training' else 'large',
        }
            
        gis.search(search_params=search_params, path_to_dir=cache_folder)
    
        # Get all files after download and identify newly downloaded ones in cache
        all_cache_files = set(os.listdir(cache_folder))
        newly_downloaded_files = list(all_cache_files - initial_cache_files)
        log(f"Actually downloaded {len(newly_downloaded_files)} new files")
        
        # Keep original filenames in cache (no GUID renaming here)
        log(f"Downloaded {len(newly_downloaded_files)} new files to cache")
        
        # Now copy all images from cache to destination using common function
        log(f"Copying images from cache to: {download_path}")
        copied_count = copy_images_from_cache_to_destination(cache_folder, download_path)

        print_summary(f"Successfully downloaded {len(newly_downloaded_files)} new images for '{actor_name}' - Images saved to: {download_path}")
        
        return True
        
    except Exception as e:
        print_error(f"Error downloading images: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download actor images from Google Image Search for StarMapr'
    )
    
    parser.add_argument('actor_name', 
                       help='Name of the actor (e.g., "Bill Murray")')
    
    
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
    success = download_actor_images(
        actor_name=args.actor_name,
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