# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StarMapr is a Python application for celebrity face recognition and detection. It uses DeepFace with the ArcFace model to create facial embeddings and perform face matching across images.

## Core Architecture

The system consists of five main components:

1. **download_celebrity_images.py** - Downloads training images from Google Image Search
   - Uses Google Custom Search API to fetch celebrity photos
   - Optimized search parameters for face portraits
   - Automatically creates celebrity folders in proper structure

2. **remove_dupe_training_images.py** - Removes near-duplicate images from celebrity folders
   - Uses perceptual hashing to identify visually similar images
   - Keeps the largest file from each duplicate group
   - Moves duplicates to a subfolder to reduce training redundancy

3. **remove_bad_training_images.py** - Removes low-quality or corrupted images from celebrity folders
   - Detects and removes images without detectable faces
   - Removes images with extremely low resolution
   - Cleans training data before embedding generation

4. **compute_average_embeddings.py** - Processes a folder of celebrity images to create average embeddings
   - Generates embeddings using DeepFace with ArcFace model
   - Computes average embedding vector for all images of a celebrity
   - Saves embeddings as pickle files (.pkl)

5. **detect_star.py** - Detects matching faces in test images using precomputed embeddings
   - Loads reference embeddings from pickle files
   - Processes test images to find matching faces
   - Extracts face crops and saves them with similarity scores
   - Uses cosine similarity for face matching

## Data Structure

- `training/` - Training data organized by celebrity name
  - Each celebrity folder contains multiple images and an average embedding file
  - Example: `training/bill_murray/` contains training images and `bill_murray_average_embedding.pkl`
- `testing/` - Test images to process for face detection
  - Organized by celebrity for validation
  - Contains `detected_headshots/` subfolder with extraction results

## Common Commands

### Download Celebrity Images
```bash
python download_celebrity_images.py "Celebrity Name" 15
```
Requires Google API credentials in .env file:
- GOOGLE_API_KEY=your_api_key_here
- GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here

### Remove Duplicate Images
```bash
python remove_dupe_training_images.py training/[celebrity_name]/
```

### Remove Bad Images
```bash
python remove_bad_training_images.py training/[celebrity_name]/
```

### Generate Average Embeddings
```bash
python compute_average_embeddings.py training/[celebrity_name]/
```

### Detect Faces in Images
```bash
python detect_star.py testing/[test_folder]/ training/[celebrity_name]/[celebrity_name]_average_embedding.pkl
```

### Custom Threshold Detection
```bash
python detect_star.py testing/[test_folder]/ training/[celebrity_name]/[celebrity_name]_average_embedding.pkl --threshold 0.7
```

## Dependencies

The project requires:
- Python 3.x
- deepface
- numpy  
- opencv-python (cv2)
- scikit-learn
- google-images-search (for image downloading)
- python-dotenv (for environment variables)
- pickle (built-in)

## Pipeline Workflow

The complete pipeline follows this sequence:

1. **Download**: Use `download_celebrity_images.py` to collect celebrity images from Google Image Search
2. **Remove Duplicates**: Run `remove_dupe_training_images.py` to eliminate near-duplicate images using perceptual hashing
3. **Remove Bad Images**: Run `remove_bad_training_images.py` to clean training data by removing corrupted or unusable images
4. **Compute**: Execute `compute_average_embeddings.py` to generate reference embeddings from cleaned training images
5. **Detect**: Use `detect_star.py` to find matching faces in test images using precomputed embeddings

### Detailed Steps

1. **Data Collection**: Use `download_celebrity_images.py` or manually collect celebrity images in `training/[name]/` folder
2. **Duplicate Removal**: Run `remove_dupe_training_images.py` to remove near-duplicate images using perceptual hashing
3. **Data Cleaning**: Run `remove_bad_training_images.py` to remove corrupted images and those without detectable faces
4. **Embedding Generation**: Run `compute_average_embeddings.py` to generate reference embeddings from cleaned training images
5. **Testing Setup**: Place test images in `testing/` folder (organized by celebrity for validation)
6. **Face Detection**: Run `detect_star.py` to find matching faces using precomputed embeddings
7. **Results Review**: Check extracted face crops in the `detected_headshots/` output folder

## Key Parameters

- Default similarity threshold: 0.6 (adjustable via --threshold)
- Supported image formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp
- Face detection model: ArcFace via DeepFace
- Similarity metric: Cosine similarity