#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pickle
from collections import Counter
from pathlib import Path
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from utils import get_actor_folder_path, get_image_files, get_average_embedding_path, load_pickle, get_env_float, print_error, print_summary, log
from utils_deepface import get_blank_embedding, get_face_embeddings
from validation import best_test_detection, load_competitors, unit

# Load environment variables
load_dotenv()

def load_embedding(embedding_path):
    """Load the precomputed average embedding from pickle file."""
    embedding = load_pickle(embedding_path)
    if embedding is None:
        raise ValueError(f"Error loading embedding file: {embedding_path}")
    log(f"Loaded embedding with shape: {embedding.shape}")
    return embedding

def detect_best_face(image_path, reference, competitors, blank, threshold=0.6, margin=0.08,
                     max_blank_similarity=0.5):
    """
    Detect faces in image and judge the one that best matches the reference.

    Returns:
        tuple: (face_data or None, decision) - face_data is set only for an accepted match
    """
    # Detect faces and get their embeddings
    face_analysis = get_face_embeddings(image_path)

    if not face_analysis:
        return None, {'status': 'no_faces'}

    return best_test_detection(face_analysis, reference, competitors, blank, threshold, margin,
                               max_blank_similarity)

def extract_face_crop(image_path, face_region, output_path):
    """Extract and save face crop from image."""
    try:
        # Read the original image
        img = cv2.imread(str(image_path))
        if img is None:
            return False

        # Extract face region
        x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
        face_crop = img[y:y+h, x:x+w]

        # Save the cropped face
        cv2.imwrite(str(output_path), face_crop)
        return True

    except Exception as e:
        print_error(f"Error extracting face crop: {e}")
        return False

def process_images(images_folder, embedding_path, threshold=0.6, output_folder="detected_headshots",
                   margin=0.08, max_blank_similarity=0.5, models_dir="04_models"):
    """
    Process all images in folder and save at most one matching face per image.
    """
    images_folder = Path(images_folder)

    if not images_folder.exists():
        raise FileNotFoundError(f"Images folder not found: {images_folder}")

    # Load reference embedding, and other actors' models that can veto ambiguous matches
    reference = unit(load_embedding(embedding_path))
    competitors = load_competitors(embedding_path, reference, models_dir)
    log(f"Competitor models: {', '.join(competitors) or 'none'}")

    # Create output folder (clear existing files first)
    output_path = images_folder / output_folder

    # Clear existing files in output folder if it exists
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)

    output_path.mkdir(exist_ok=True)

    # Get all image files
    image_files = get_image_files(images_folder, exclude_subdirs=True)

    if not image_files:
        log(f"No image files found in {images_folder}")
        return

    blank = get_blank_embedding()
    log(f"Processing {len(image_files)} images...")

    total_detections = 0
    rejections = Counter()

    for img_file in image_files:
        log(f"Processing: {img_file.name}")

        face, decision = detect_best_face(img_file, reference, competitors, blank, threshold, margin,
                                          max_blank_similarity)
        rejections['low_information_faces'] += decision.get('low_information_faces', 0)

        if face is None:
            rejections[decision['status']] += 1
            log(f"  → No reliable match ({decision['status']})")
            continue

        # Create output filename
        output_filename = f"{img_file.stem}_{decision['score']:.3f}.jpg"
        output_file_path = output_path / output_filename

        # Extract and save face crop
        if extract_face_crop(img_file, face['bounding_box'], output_file_path):
            log(f"  → Detected face with similarity {decision['score']:.3f} → {output_filename}")
            total_detections += 1
        else:
            log(f"  → Failed to extract face crop")

    # Summary
    log(f"\nDetection Summary:")
    log(f"Images processed: {len(image_files)}")
    log(f"Images with detections: {total_detections}")
    log(f"Rejections: {dict(rejections)}")
    log(f"Output folder: {output_path}")

    if total_detections == 0:
        print_error("No faces matching the reference were detected across all images.")
    else:
        print_summary(f"Face detection completed! Found {total_detections} matching faces across {len(image_files)} images.")

def main():
    parser = argparse.ArgumentParser(description='Detect star faces in images using precomputed embeddings')
    parser.add_argument('actor_name', help='Actor name (e.g., "Bill Murray")')
    # Get default threshold from environment variable
    default_threshold = get_env_float('TESTING_DETECTION_THRESHOLD', 0.6)
    parser.add_argument('--threshold', '-t', type=float, default=default_threshold,
                       help=f'Similarity threshold for face matching (default: {default_threshold})')
    parser.add_argument('--output', '-o', default='detected_headshots',
                       help='Output folder name (default: detected_headshots)')

    args = parser.parse_args()

    try:
        # Construct paths automatically
        images_folder = get_actor_folder_path(args.actor_name, 'testing')
        embedding_file = get_average_embedding_path(args.actor_name, 'training')

        # Verify paths exist
        if not os.path.exists(images_folder):
            raise FileNotFoundError(f"Testing folder not found: {images_folder}")
        if not os.path.exists(embedding_file):
            raise FileNotFoundError(f"Embedding file not found: {embedding_file}")

        log(f"Using testing folder: {images_folder}")
        log(f"Using embedding file: {embedding_file}")

        # Same competitor margin as video headshot extraction
        process_images(images_folder, embedding_file, args.threshold, args.output,
                       margin=get_env_float('OPERATIONS_MIN_MATCH_MARGIN', 0.08),
                       max_blank_similarity=get_env_float('MAX_BLANK_SIMILARITY', 0.5))

    except Exception as e:
        print_error(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
