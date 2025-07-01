#!/usr/bin/env python3
"""
Generic Video Downloader

Downloads videos from various sites (YouTube, Vimeo, TikTok, etc.) using yt-dlp
and saves them to videos/[site]_[video_id]/ folder.

Requires yt-dlp: pip install yt-dlp

Usage:
    python3 download_video.py "https://www.youtube.com/watch?v=VIDEO_ID"
    python3 download_video.py "https://vimeo.com/123456789"
    python3 download_video.py "https://www.tiktok.com/@user/video/1234567890"
"""

import argparse
import os
import sys
import re
import subprocess
import json
from pathlib import Path
from urllib.parse import urlparse


def extract_site_and_id(url):
    """Extract site name and video ID from URL."""
    try:
        # Use yt-dlp to extract info without downloading
        cmd = ['yt-dlp', '--dump-json', '--no-download', url]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout.split('\n')[0])
        
        site = info.get('extractor_key', 'unknown').lower()
        video_id = info.get('id', 'unknown')
        title = info.get('title', 'untitled')
        
        # Clean up common site names
        site_mapping = {
            'youtube': 'youtube',
            'vimeo': 'vimeo',
            'tiktok': 'tiktok',
            'twitter': 'twitter',
            'instagram': 'instagram',
            'facebook': 'facebook',
            'dailymotion': 'dailymotion',
            'twitch': 'twitch'
        }
        
        site = site_mapping.get(site, site)
        
        return site, video_id, title
        
    except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError) as e:
        # Fallback: try to parse URL manually
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        if 'youtube.com' in domain or 'youtu.be' in domain:
            match = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
            return 'youtube', match.group(1) if match else 'unknown', 'video'
        elif 'vimeo.com' in domain:
            match = re.search(r'vimeo\.com/(\d+)', url)
            return 'vimeo', match.group(1) if match else 'unknown', 'video'
        elif 'tiktok.com' in domain:
            match = re.search(r'/video/(\d+)', url)
            return 'tiktok', match.group(1) if match else 'unknown', 'video'
        else:
            # Generic fallback
            site = domain.replace('www.', '').split('.')[0]
            return site, 'unknown', 'video'


def check_ytdlp():
    """Check if yt-dlp is installed."""
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def sanitize_filename(filename):
    """Remove invalid characters from filename."""
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def download_video(video_url, site, video_id, title):
    """Download video to videos/[site]_[video_id]/ folder."""
    videos_dir = Path("videos")
    folder_name = f"{site}_{video_id}"
    video_dir = videos_dir / folder_name
    
    # Create directories
    video_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {site} video '{title}' to {video_dir}/")
    
    # Sanitize title for filename
    safe_title = sanitize_filename(title)
    
    # yt-dlp command to download video
    cmd = [
        'yt-dlp',
        '-o', str(video_dir / f'{safe_title}.%(ext)s'),
        '--format', 'best[height<=1080]',  # Download best quality up to 1080p
        '--write-info-json',  # Save metadata
        '--write-thumbnail',  # Save thumbnail
        '--write-description',  # Save description if available
        video_url
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Successfully downloaded video to {video_dir}/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading video: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download video from various sites to videos/[site]_[video_id]/ folder')
    parser.add_argument('url', help='Video URL from supported site')
    parser.add_argument('--format', default='best[height<=1080]', 
                       help='Video format (default: best[height<=1080])')
    parser.add_argument('--list-extractors', action='store_true',
                       help='List all supported sites/extractors')
    
    args = parser.parse_args()
    
    # Check if yt-dlp is installed
    if not check_ytdlp():
        print("Error: yt-dlp is not installed.")
        print("Install it with: pip install yt-dlp")
        sys.exit(1)
    
    # List extractors if requested
    if args.list_extractors:
        try:
            result = subprocess.run(['yt-dlp', '--list-extractors'], 
                                  capture_output=True, text=True, check=True)
            print("Supported sites/extractors:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error listing extractors: {e}")
        return
    
    try:
        # Extract site and video info
        site, video_id, title = extract_site_and_id(args.url)
        print(f"Site: {site}")
        print(f"Video ID: {video_id}")
        print(f"Title: {title}")
        
        # Download video
        success = download_video(args.url, site, video_id, title)
        
        if success:
            print(f"Video downloaded successfully to videos/{site}_{video_id}/")
        else:
            print("Failed to download video")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDownload cancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    main()