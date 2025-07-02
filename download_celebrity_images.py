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
    
    Downloads 10 general images and 10 show-specific images (20 total per page).
    Increasing page number pulls the next 20 images for more training data.
    
    Args:
        celebrity_name (str): Name of the celebrity to search for
        mode (str): 'training' for solo portraits or 'testing' for group photos
        show (str): Show name to include in search for targeted results
        page (int): Page number for pagination (default: 1)
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
    
    def create_search_params(query, num_imgs, start_index):
        """Create search parameters based on mode"""
        if mode == 'training':
            return {
                'q': query,
                'num': num_imgs,
                'start': start_index,
                'fileType': 'jpg|jpeg|png',
                'safe': 'medium',
                'imgType': 'face',
                'imgSize': 'medium'
            }
        else:  # testing mode
            return {
                'q': query,
                'num': num_imgs,
                'start': start_index,
                'fileType': 'jpg|jpeg|png',
                'safe': 'medium',
                'imgType': 'photo',
                'imgSize': 'large'
            }
    
    query_suffix = 'face' if mode == 'training' else 'group'
    
    # Always download 10 general + 10 show-specific images (20 total)
    images_per_search = 10
    
    # Calculate start index for pagination (each page = 20 images total)
    start_index = (page - 1) * images_per_search + 1
    
    # Create search queries
    searches = []
    searches.append((f'{celebrity_name} {query_suffix}', images_per_search, start_index, 'general'))
    searches.append((f'{celebrity_name} {show} {query_suffix}', images_per_search, start_index, f'show-specific ({show})'))
    
    print(f"Downloading page {page} images for '{celebrity_name}' (10 general + 10 {show}-specific = 20 total)...")
    
    # Get initial file count to track new downloads
    initial_files = set(os.listdir(download_path))
    
    try:
        # Perform searches and downloads
        for query, num_imgs, start_idx, search_type in searches:
            print(f"  Downloading {num_imgs} {search_type} images...")
            search_params = create_search_params(query, num_imgs, start_idx)
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
                       help='Page number for pagination (default: 1). Each page downloads 20 images.')
    
    args = parser.parse_args()
    
    # Validate input
    if args.page <= 0:
        print_error("Page number must be positive")
        sys.exit(1)
    
    if args.page > 10:
        print("Warning: High page number requested. Google API has daily limits.")
    
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