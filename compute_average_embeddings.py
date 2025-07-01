#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pickle
from deepface import DeepFace
from pathlib import Path

def compute_average_embeddings(folder_path):
    """
    Compute average embeddings for all images in a folder using DeepFace with ArcFace.
    
    Args:
        folder_path (str): Path to folder containing celebrity images
        
    Returns:
        numpy.ndarray: Average embedding vector
    """
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in folder_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        raise ValueError(f"No image files found in {folder_path}")
    
    print(f"Found {len(image_files)} images in {folder_path}")
    
    embeddings = []
    successful_embeddings = 0
    
    for img_file in image_files:
        try:
            print(f"Processing: {img_file.name}")
            # Generate embedding using DeepFace with ArcFace model
            embedding = DeepFace.represent(str(img_file), model_name='ArcFace')
            embeddings.append(embedding[0]['embedding'])
            successful_embeddings += 1
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")
            continue
    
    if not embeddings:
        raise ValueError("No embeddings could be generated from the images")
    
    print(f"Successfully processed {successful_embeddings}/{len(image_files)} images")
    
    # Convert to numpy array and compute average
    embeddings_array = np.array(embeddings)
    average_embedding = np.mean(embeddings_array, axis=0)
    
    return average_embedding

def save_embedding(embedding, output_path):
    """Save embedding to a pickle file."""
    with open(output_path, 'wb') as f:
        pickle.dump(embedding, f)
    print(f"Average embedding saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compute average embeddings for celebrity images using ArcFace')
    parser.add_argument('celebrity_name', help='Celebrity name (will use training/celebrity_name/ folder)')
    parser.add_argument('--output', '-o', help='Output file path for the average embedding', 
                       default=None)
    
    args = parser.parse_args()
    
    try:
        # Convert celebrity name to folder path
        celebrity_folder = args.celebrity_name.lower().replace(' ', '_')
        folder_path = Path(f"training/{celebrity_folder}")
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Training folder not found: {folder_path}")
        
        # Compute average embedding
        avg_embedding = compute_average_embeddings(folder_path)
        
        # Generate output filename if not provided
        if args.output is None:
            args.output = folder_path / f"{celebrity_folder}_average_embedding.pkl"
        
        # Save the embedding
        save_embedding(avg_embedding, args.output)
        
        print(f"Average embedding shape: {avg_embedding.shape}")
        print("Process completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()