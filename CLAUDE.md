# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StarMapr is a Python application for celebrity face recognition and detection. It uses DeepFace with the ArcFace model to create facial embeddings and perform face matching across images.

## Core Architecture

The system follows a **sequential pipeline architecture** with five main stages that process celebrity images from raw downloads to face detection:

**Pipeline Flow**: Data Collection → Duplicate Removal → Data Cleaning → Face Consistency Validation → Embedding Generation → Face Detection

The system consists of seven main components:

1. **run_pipeline.py** - Interactive pipeline runner for streamlined workflow execution
   - Provides numbered menu of all pipeline steps
   - Automatic path validation and error checking
   - Guided celebrity name input and image count selection
   - Seamless execution of training and testing workflows

2. **download_celebrity_images.py** - Downloads training images from Google Image Search
   - Uses Google Custom Search API to fetch celebrity photos
   - Optimized search parameters for face portraits
   - Automatically creates celebrity folders in proper structure
   - Names images using first 8 characters of GUIDs to prevent collisions on reruns, allowing safe duplicate removal and bad image handling

3. **remove_dupe_training_images.py** - Removes near-duplicate images from celebrity folders
   - Uses perceptual hashing to identify visually similar images
   - Keeps the largest file from each duplicate group
   - Moves duplicates to a subfolder to reduce training redundancy

4. **remove_bad_training_images.py** - Removes images that don't meet face count requirements
   - Training mode: Removes images without exactly 1 face
   - Testing mode: Removes images with fewer than 4 or more than 10 faces
   - Uses DeepFace for accurate face detection and validation

5. **remove_face_outliers.py** - Removes face outliers from celebrity training images
   - Compares all faces using DeepFace ArcFace embeddings and cosine similarity
   - Identifies faces that are significantly different from the majority group
   - Moves outlier faces to an `outliers/` subfolder
   - Helps ensure training data consistency by removing incorrect celebrity faces

6. **compute_average_embeddings.py** - Processes a folder of celebrity images to create average embeddings
   - Generates embeddings using DeepFace with ArcFace model
   - Computes average embedding vector for all images of a celebrity
   - Saves embeddings as pickle files (.pkl)

7. **eval_star_detection.py** - Detects matching faces in test images using precomputed embeddings
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

## Dependencies Installation

```bash
pip install deepface numpy opencv-python scikit-learn google-images-search python-dotenv
```

## Common Commands

### Interactive Pipeline Runner (Recommended)
```bash
python3 run_pipeline.py
```
Launches an interactive menu that guides you through the complete pipeline process for a celebrity. The script automatically handles path management, validates prerequisites, and provides numbered options for each step.

### Manual Commands
All scripts now use `--training` and `--testing` flags to specify the dataset type and automatically handle folder paths.

### Download Celebrity Images
```bash
# For training dataset (solo portraits)
python3 download_celebrity_images.py "Celebrity Name" 15 --training

# For testing dataset (group photos)
python3 download_celebrity_images.py "Celebrity Name" 10 --testing
```
Requires Google API credentials in .env file:
- GOOGLE_API_KEY=your_api_key_here
- GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here

### Remove Duplicate Images
```bash
# Training dataset
python3 remove_dupe_training_images.py --training "Celebrity Name"

# Testing dataset
python3 remove_dupe_training_images.py --testing "Celebrity Name"
```

### Remove Bad Images
```bash
# Training dataset (keeps images with exactly 1 face)
python3 remove_bad_training_images.py --training "Celebrity Name"

# Testing dataset (keeps images with 4-10 faces)
python3 remove_bad_training_images.py --testing "Celebrity Name"
```

### Remove Face Outliers
```bash
# Training dataset (removes faces inconsistent with majority)
python3 remove_face_outliers.py --training "Celebrity Name"

# Testing dataset
python3 remove_face_outliers.py --testing "Celebrity Name"

# Custom similarity threshold (default: 0.1)
python3 remove_face_outliers.py --training "Celebrity Name" --threshold 0.2
```

### Generate Average Embeddings
```bash
python3 compute_average_embeddings.py training/[celebrity_name]/
```

### Detect Faces in Images
```bash
python3 eval_star_detection.py testing/[test_folder]/ training/[celebrity_name]/[celebrity_name]_average_embedding.pkl
```

### Custom Threshold Detection
```bash
python3 eval_star_detection.py testing/[test_folder]/ training/[celebrity_name]/[celebrity_name]_average_embedding.pkl --threshold 0.7
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

The complete pipeline follows this sequence. You can use the interactive pipeline runner (`python3 run_pipeline.py`) for guided execution, or run commands manually:

### Training Pipeline (Solo Portraits)
1. **Download Training Data**: `python3 download_celebrity_images.py "Celebrity Name" 15 --training`
2. **Remove Duplicates**: `python3 remove_dupe_training_images.py --training "Celebrity Name"`
3. **Remove Bad Images**: `python3 remove_bad_training_images.py --training "Celebrity Name"` (keeps exactly 1 face)
4. **Remove Face Outliers**: `python3 remove_face_outliers.py --training "Celebrity Name"` (removes inconsistent faces)
5. **Generate Embeddings**: `python3 compute_average_embeddings.py training/celebrity_name/`

### Testing Pipeline (Group Photos)
1. **Download Test Data**: `python3 download_celebrity_images.py "Celebrity Name" 10 --testing`
2. **Remove Duplicates**: `python3 remove_dupe_training_images.py --testing "Celebrity Name"`
3. **Remove Bad Images**: `python3 remove_bad_training_images.py --testing "Celebrity Name"` (keeps 4-10 faces)
4. **Run Detection**: `python3 eval_star_detection.py testing/celebrity_name/ training/celebrity_name/celebrity_name_average_embedding.pkl`

### Complete Example Workflow
```bash
# 1. Training phase
python3 download_celebrity_images.py "Bill Murray" 20 --training
python3 remove_dupe_training_images.py --training "Bill Murray"
python3 remove_bad_training_images.py --training "Bill Murray"
python3 remove_face_outliers.py --training "Bill Murray"
python3 compute_average_embeddings.py training/bill_murray/

# 2. Testing phase
python3 download_celebrity_images.py "Bill Murray" 15 --testing
python3 remove_dupe_training_images.py --testing "Bill Murray"
python3 remove_bad_training_images.py --testing "Bill Murray"
python3 eval_star_detection.py testing/bill_murray/ training/bill_murray/bill_murray_average_embedding.pkl
```

## Key Parameters

- **Face Detection Model**: ArcFace via DeepFace
- **Similarity Metric**: Cosine similarity
- **Default Detection Threshold**: 0.6 (adjustable via --threshold)
- **Supported Image Formats**: .jpg, .jpeg, .png, .bmp, .tiff, .webp
- **Training Face Count**: Exactly 1 face required
- **Testing Face Count**: 4-10 faces required
- **Duplicate Detection Threshold**: 5 Hamming distance (adjustable via --threshold)
- **Face Outlier Detection Threshold**: 0.1 cosine similarity (adjustable via --threshold)

## Architecture Notes

- **GUID-based Naming**: Image files use first 8 characters of GUIDs to prevent filename collisions
- **Automatic Folder Management**: Scripts create `bad/`, `duplicates/`, and `outliers/` subfolders automatically
- **Dry-run Support**: Most scripts support `--dry-run` flag for safe testing
- **Error Handling**: Comprehensive exception handling with detailed user feedback
- **Modular Design**: Each script handles a single responsibility in the pipeline