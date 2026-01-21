#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pickle
from pathlib import Path
from utils import get_actor_folder_path, get_image_files, get_average_embedding_path, save_pickle, print_error, print_summary, log
from utils_deepface import get_face_embeddings

def compute_average_embeddings(folder_path):
    """
    Compute average embeddings for all images in a folder using DeepFace with ArcFace.
    
    Args:
        folder_path (str): Path to folder containing actor images
        
    Returns:
        tuple: (average_embedding, successful_count) - Average embedding vector and count of successful embeddings
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Get all image files
    image_files = get_image_files(folder_path, exclude_subdirs=True)
    
    if not image_files:
        raise ValueError(f"No image files found in {folder_path}")
    
    log(f"Found {len(image_files)} images in {folder_path}")
    
    embeddings = []
    successful_embeddings = 0
    
    for img_file in image_files:
        log(f"Processing: {img_file.name}")
        face_embeddings = get_face_embeddings(img_file)
        if face_embeddings:
            embeddings.append(face_embeddings[0]['embedding'])
            successful_embeddings += 1
    
    if not embeddings:
        raise ValueError("No embeddings could be generated from the images")
    
    log(f"Successfully processed {successful_embeddings}/{len(image_files)} images")
    
    # Convert to numpy array and compute average
    embeddings_array = np.array(embeddings)
    average_embedding = np.mean(embeddings_array, axis=0)
    
    return average_embedding, successful_embeddings

def save_embedding(embedding, output_path):
    """Save embedding to a pickle file."""
    if save_pickle(embedding, output_path):
        log(f"Average embedding saved to: {output_path}")
    else:
        raise Exception(f"Failed to save embedding to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compute average embeddings for actor images using ArcFace')
    parser.add_argument('actor_name', help='Actor name (will use training/actor_name/ folder)')
    parser.add_argument('--output', '-o', help='Output file path for the average embedding', 
                       default=None)
    
    args = parser.parse_args()
    
    try:
        # Convert actor name to folder path
        folder_path = Path(get_actor_folder_path(args.actor_name, 'training'))
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Training folder not found: {folder_path}")
        
        # Compute average embedding
        avg_embedding, successful_count = compute_average_embeddings(folder_path)
        
        # Generate output filename if not provided
        if args.output is None:
            args.output = get_average_embedding_path(args.actor_name, 'training')
        
        # Save the embedding
        save_embedding(avg_embedding, args.output)
        
        log(f"Average embedding shape: {avg_embedding.shape}")
        print_summary(f"Successfully computed average embedding for {args.actor_name} from {successful_count} images.")
        
    except Exception as e:
        print_error(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
