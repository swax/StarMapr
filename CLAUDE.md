# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StarMapr is a Python application for celebrity face recognition and detection. It uses DeepFace with the ArcFace model to create facial embeddings and perform face matching across images.

## Core Architecture

The system follows a **sequential pipeline architecture** with five main stages that process celebrity images from raw downloads to face detection:

**Pipeline Flow**: Data Collection → Duplicate Removal → Data Cleaning → Face Consistency Validation → Embedding Generation → Face Detection

The system consists of eleven main components:

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

8. **download_video.py** - Downloads videos from various platforms for processing
   - Downloads videos from YouTube, Vimeo, TikTok, and other supported sites using yt-dlp
   - Saves videos to `videos/[site]_[video_id]/` folder structure
   - Includes metadata extraction (title, description, thumbnail)
   - Supports format selection and quality control (default: best up to 1080p)

9. **extract_video_frames.py** - Extracts frames from videos using binary search pattern
   - Takes video folder path and automatically finds video file within folder
   - Uses binary search pattern to extract representative frames from videos
   - Creates frames in a `frames/` subfolder within the video folder
   - Allows specifying the number of frames to extract
   - Names frames with zero-padded frame numbers (e.g., `00000123.jpg`)
   - Skips existing frames to allow resuming interrupted extractions

10. **extract_frame_faces.py** - Detects faces in extracted video frames
    - Takes video folder path and automatically processes frames in `frames/` subfolder
    - Processes all frames to detect faces using DeepFace ArcFace
    - Saves face detection data with bounding boxes and embeddings to pickle files
    - Creates `.pkl` files alongside each frame image with face metadata
    - Supports multiple faces per frame with individual face IDs
    - Enables face tracking and analysis across video sequences

11. **extract_video_headshots.py** - Extracts celebrity headshots from video frames
    - Takes celebrity name and video folder path as parameters
    - Loads celebrity reference embeddings and scans frame face data
    - Calculates cosine similarity between detected faces and reference celebrity
    - Selects top 5 most similar faces and extracts cropped headshots
    - Saves headshots with similarity scores and frame information

## Data Structure

- `training/` - Training data organized by celebrity name
  - Each celebrity folder contains multiple images and an average embedding file
  - Example: `training/bill_murray/` contains training images and `bill_murray_average_embedding.pkl`
- `testing/` - Test images to process for face detection
  - Organized by celebrity for validation
  - Contains `detected_headshots/` subfolder with extraction results
- `videos/` - Downloaded videos organized by source and video ID
  - Each video folder contains the video file, metadata, and extracted frames
  - Example: `videos/youtube_ABC123/` contains video file and `frames/` subfolder
  - Frame data includes both image files and corresponding face detection pickle files
  - Extracted headshots are saved in `headshots/` subfolder with similarity scores

## Dependencies Installation

```bash
pip install deepface numpy opencv-python scikit-learn google-images-search python-dotenv yt-dlp
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
# Uses default count from TRAINING_IMAGE_COUNT/TESTING_IMAGE_COUNT in .env
python3 download_celebrity_images.py "Celebrity Name" --training
python3 download_celebrity_images.py "Celebrity Name" --testing

# Or specify custom count
python3 download_celebrity_images.py "Celebrity Name" 15 --training
python3 download_celebrity_images.py "Celebrity Name" 10 --testing
```
Requires configuration in .env file:
- GOOGLE_API_KEY=your_api_key_here
- GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here
- TRAINING_IMAGE_COUNT=20
- TESTING_IMAGE_COUNT=30

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

# Custom similarity threshold (default from TRAINING_OUTLIER_THRESHOLD)
python3 remove_face_outliers.py --training "Celebrity Name" --threshold 0.2
```

### Generate Average Embeddings
```bash
python3 compute_average_embeddings.py "Celebrity Name"
```

### Detect Faces in Images
```bash
python3 eval_star_detection.py "Celebrity Name"
```

### Custom Threshold Detection
```bash
# Custom threshold (default from TESTING_DETECTION_THRESHOLD)
python3 eval_star_detection.py "Celebrity Name" --threshold 0.7
```

### Video Processing Pipeline

#### Download Videos
```bash
# Download from any supported platform (YouTube, Vimeo, TikTok, etc.)
python3 download_video.py "https://www.youtube.com/watch?v=VIDEO_ID"
python3 download_video.py "https://vimeo.com/123456789"

# List all supported video platforms
python3 download_video.py --list-extractors
```

#### Extract Frames from Video
```bash
# Extract frames using default count from OPERATIONS_EXTRACT_FRAME_COUNT
python3 extract_video_frames.py videos/youtube_ABC123/

# Or specify custom frame count
python3 extract_video_frames.py videos/youtube_ABC123/ 50

# Dry run to see what frames would be extracted
python3 extract_video_frames.py videos/youtube_ABC123/ 50 --dry-run
```

#### Extract Faces from Video Frames
```bash
# Process all frames (script automatically uses frames/ subfolder)
python3 extract_frame_faces.py videos/youtube_ABC123/

# Dry run to see what would be processed
python3 extract_frame_faces.py videos/youtube_ABC123/ --dry-run
```

#### Extract Celebrity Headshots from Video
```bash
# Extract top 5 headshots for a celebrity from video frames
python3 extract_video_headshots.py "Bill Murray" videos/youtube_ABC123/

# Custom similarity threshold (default from OPERATIONS_HEADSHOT_MATCH_THRESHOLD)
python3 extract_video_headshots.py "Bill Murray" videos/youtube_ABC123/ --threshold 0.7

# Dry run to see what would be extracted
python3 extract_video_headshots.py "Bill Murray" videos/youtube_ABC123/ --dry-run
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
- yt-dlp (for video downloading)
- pickle (built-in)

## Pipeline Workflow

The complete pipeline follows this sequence. You can use the interactive pipeline runner (`python3 run_pipeline.py`) for guided execution, or run commands manually:

### Training Pipeline (Solo Portraits)
1. **Download Training Data**: `python3 download_celebrity_images.py "Celebrity Name" --training`
2. **Remove Duplicates**: `python3 remove_dupe_training_images.py --training "Celebrity Name"`
3. **Remove Bad Images**: `python3 remove_bad_training_images.py --training "Celebrity Name"` (keeps exactly 1 face)
4. **Remove Face Outliers**: `python3 remove_face_outliers.py --training "Celebrity Name"` (removes inconsistent faces)
5. **Generate Embeddings**: `python3 compute_average_embeddings.py "Celebrity Name"`

### Testing Pipeline (Group Photos)
1. **Download Test Data**: `python3 download_celebrity_images.py "Celebrity Name" --testing`
2. **Remove Duplicates**: `python3 remove_dupe_training_images.py --testing "Celebrity Name"`
3. **Remove Bad Images**: `python3 remove_bad_training_images.py --testing "Celebrity Name"` (keeps 4-10 faces)
4. **Run Detection**: `python3 eval_star_detection.py "Celebrity Name"`


### Video Processing Pipeline
1. **Download Video**: `python3 download_video.py "https://youtube.com/watch?v=VIDEO_ID"`
2. **Extract Frames**: `python3 extract_video_frames.py videos/youtube_VIDEO_ID/` (uses default frame count)
3. **Extract Faces**: `python3 extract_frame_faces.py videos/youtube_VIDEO_ID/`
4. **Extract Celebrity Headshots**: `python3 extract_video_headshots.py "Celebrity Name" videos/youtube_VIDEO_ID/`


## Key Parameters

All default values configurable via environment variables in .env file:

- **Face Detection Model**: ArcFace via DeepFace
- **Similarity Metric**: Cosine similarity
- **Supported Image Formats**: .jpg, .jpeg, .png, .bmp, .tiff, .webp
- **Training Face Count**: Exactly 1 face required
- **Testing Face Count**: 4-10 faces required

### Environment Variables:
- **TRAINING_DUPLICATE_THRESHOLD**: 5 (Hamming distance, 0-64, adjustable via --threshold)
- **TRAINING_OUTLIER_THRESHOLD**: 0.1 (cosine similarity, 0.0-1.0, adjustable via --threshold)  
- **TESTING_DETECTION_THRESHOLD**: 0.6 (cosine similarity, 0.0-1.0, adjustable via --threshold)
- **OPERATIONS_EXTRACT_FRAME_COUNT**: 50 (number of frames to extract from videos)
- **OPERATIONS_HEADSHOT_MATCH_THRESHOLD**: 0.6 (cosine similarity, 0.0-1.0, adjustable via --threshold)
- **TRAINING_IMAGE_COUNT**: 20 (default training images to download)
- **TESTING_IMAGE_COUNT**: 30 (default testing images to download)

### Other Parameters:
- **Video Headshot Extraction**: Top 5 most similar faces with 10% padding around face region

## Architecture Notes

- **GUID-based Naming**: Image files use first 8 characters of GUIDs to prevent filename collisions
- **Automatic Folder Management**: Scripts create `bad/`, `duplicates/`, and `outliers/` subfolders automatically
- **Dry-run Support**: Most scripts support `--dry-run` flag for safe testing
- **Error Handling**: Comprehensive exception handling with detailed user feedback
- **Modular Design**: Each script handles a single responsibility in the pipeline