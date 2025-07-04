# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StarMapr is a Python application for celebrity face recognition and detection using DeepFace with the ArcFace model.

## Core Architecture

The system follows a **three-tier hierarchical architecture**:

**Top Level**: `run_headshot_detection.py` - Complete end-to-end workflow
**Mid Level**: `run_celebrity_training.py` - Automated celebrity training + testing
**Low Level**: `run_pipeline_steps.py` - Manual step-by-step execution

**Pipeline Flow**: Training → Testing → Operations

The system consists of 17 components organized in three execution tiers:

### Top-Level Orchestration (Complete Workflow)

1. **run_headshot_detection.py** - **PRIMARY ENTRY POINT**
   - Takes video URL and celebrities as input
   - Calls `run_celebrity_training.py` for each celebrity
   - Executes full operations pipeline (download → frames → faces → headshots)
   - Uses adaptive frame extraction if no headshots found

### Mid-Level Automation (Training + Testing)

2. **run_celebrity_training.py** - Automated celebrity training + testing
   - Runs complete training pipeline (steps 1-5)
   - Runs complete testing pipeline (steps 6-10)
   - Iteratively downloads until thresholds met (15+ training, 4+ detected headshots)
   - Copies final model to models directory

3. **run_pipeline_steps.py** - Manual pipeline runner
   - Interactive numbered menu of all 15 pipeline steps
   - Manual step-by-step execution of training pipeline
   - Used for debugging and manual control

### TRAINING PIPELINE (Steps 1-5)

1. **download_celebrity_images.py** - Downloads from Google Image Search
   - 20 images per page, each page uses different keywords
   - Training: different keywords for more face variety
   - Testing: keywords targeting group photos
   - GUID-based naming prevents collisions

2. **remove_dupe_training_images.py** - Removes near-duplicates
   - Perceptual hashing, keeps largest file

3. **remove_bad_training_images.py** - Face count validation
   - Training: exactly 1 face | Testing: 3-10 faces

4. **remove_face_outliers.py** - Removes inconsistent faces
   - Cosine similarity comparison, moves outliers to subfolder

5. **compute_average_embeddings.py** - Creates celebrity embeddings
   - Generates and averages ArcFace embeddings, saves as .pkl

### TESTING PIPELINE (Steps 6-10)

6. **download_celebrity_images.py** - Downloads test images (same as step 1)
7. **remove_dupe_training_images.py** - Removes test duplicates (same as step 2)
8. **remove_bad_training_images.py** - Test face validation (keeps 3-10 people for group testing)
9. **eval_star_detection.py** - Detects matching faces in test images
   - Loads reference embeddings, extracts matching face crops

10. **Accept Model** - Copies embedding to models directory

### OPERATIONS PIPELINE (Steps 11-14)

11. **download_video.py** - Downloads videos using yt-dlp
    - Supports YouTube, Vimeo, TikTok, saves to `videos/[site]_[id]/`

12. **extract_video_frames.py** - Extracts frames using binary search
    - Saves to `frames/` subfolder with zero-padded naming

13. **extract_frame_faces.py** - Detects faces in frames
    - Creates .pkl files with face data alongside each frame

14. **extract_video_headshots.py** - Extracts celebrity headshots
    - Matches against reference embeddings, saves top 5 matches

### UTILITY MODULES

15. **utils.py** - Common functions (path conversion, argument parsing, file operations)
16. **utils_deepface.py** - DeepFace utilities with caching
17. **print_pkl.py** - Pickle file inspector
18. **run_integration_test.py** - Integration test script with mock data

## Data Structure

- `training/[celebrity]/` - Training images + `[celebrity]_average_embedding.pkl`
- `testing/[celebrity]/` - Test images + `detected_headshots/` subfolder  
- `models/[celebrity]/` - Final accepted models
- `videos/[site]_[id]/` - Video file + `frames/` + `headshots/` subfolders

## Usage

### Primary Entry Point
```bash
python3 run_headshot_detection.py "https://youtube.com/watch?v=VIDEO_ID" --show "SNL" "Bill Murray" "Tina Fey"
```

### Mid-Level Training
```bash
python3 run_celebrity_training.py "Celebrity Name" "Show Name"
```

### Manual Control
```bash
python3 run_pipeline_steps.py
```

### Integration Testing
```bash
# Unzip mock data first
unzip mocks.zip

# Run integration test
python3 run_integration_test.py
```

### Manual Commands
```bash
# Training pipeline (steps 1-5)
python3 download_celebrity_images.py "Name" --training --show "Show" --page 1
python3 remove_dupe_training_images.py --training "Name"
python3 remove_bad_training_images.py --training "Name"
python3 remove_face_outliers.py --training "Name"
python3 compute_average_embeddings.py "Name"

# Testing pipeline (steps 6-10)
python3 download_celebrity_images.py "Name" --testing --show "Show"
python3 remove_dupe_training_images.py --testing "Name"
python3 remove_bad_training_images.py --testing "Name"
python3 eval_star_detection.py "Name"

# Operations pipeline (steps 11-14)
python3 download_video.py "https://youtube.com/watch?v=VIDEO_ID"
python3 extract_video_frames.py videos/youtube_ABC123/
python3 extract_frame_faces.py videos/youtube_ABC123/
python3 extract_video_headshots.py "Name" videos/youtube_ABC123/
```

## Dependencies
```bash
pip install deepface numpy opencv-python scikit-learn google-images-search python-dotenv yt-dlp
```

## Pipeline Stages

**Training (Steps 1-5)**: Download → Remove Dupes → Remove Bad → Remove Outliers → Generate Embeddings
**Testing (Steps 6-10)**: Download → Remove Dupes → Remove Bad → Detect Faces → Accept Model  
**Operations (Steps 11-14)**: Download Video → Extract Frames → Extract Faces → Extract Headshots


## Configuration

### Environment Variables (.env)
- **TRAINING_DUPLICATE_THRESHOLD**: 5 (Hamming distance)
- **TRAINING_OUTLIER_THRESHOLD**: 0.2 (cosine similarity)
- **TESTING_DETECTION_THRESHOLD**: 0.4 (cosine similarity)
- **OPERATIONS_EXTRACT_FRAME_COUNT**: 50
- **OPERATIONS_HEADSHOT_MATCH_THRESHOLD**: 0.4
- **MIN_FACE_SIZE**: 50 (pixels)
- **TRAINING_MIN_IMAGES**: 15 (minimum training images required)
- **TESTING_MIN_HEADSHOTS**: 4 (minimum detected headshots required)
- **MAX_DOWNLOAD_PAGES**: 5 (maximum pages to download)
- **GOOGLE_API_KEY**: your_api_key_here
- **GOOGLE_SEARCH_ENGINE_ID**: your_search_engine_id_here

### Key Parameters
- **Model**: ArcFace via DeepFace
- **Similarity**: Cosine similarity
- **Training Face Count**: Exactly 1 (solo portraits)
- **Testing Face Count**: 3-10 (group photos for embedding validation)
- **Headshot Extraction**: Top 5 matches