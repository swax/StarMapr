# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StarMapr is a Python application for actor face recognition and detection using DeepFace with the ArcFace model.

## Core Architecture

The system follows a **three-tier hierarchical architecture**:

**Top Level**: `run_headshot_detection.py` - Complete end-to-end workflow
**Mid Level**: `run_actor_training.py` - Automated actor training + testing
**Low Level**: `run_pipeline_steps.py` - Manual step-by-step execution

**Pipeline Flow**: Training → Testing → Operations

### Architecture Diagram

```
run_integration_test.py             # Integration test root
└── run_headshot_detection.py       # ★ PRIMARY ENTRY POINT
    ├── run_actor_training.py   # ★ MID-LEVEL automation
    │   ├── download_actor_images.py
    │   ├── remove_dupe_training_images.py
    │   ├── remove_bad_training_images.py
    │   ├── remove_face_outliers.py
    │   ├── compute_average_embeddings.py
    │   └── eval_star_detection.py
    ├── download_video.py
    ├── extract_video_frames.py
    ├── extract_frame_faces.py
    ├── extract_video_headshots.py
    └── extract_video_thumbnail.py
```

The system consists of 19 components organized in three execution tiers:

### Top-Level Orchestration (Complete Workflow)

1. **run_headshot_detection.py** - **PRIMARY ENTRY POINT**
   - Takes video URL and actors as input
   - Calls `run_actor_training.py` for each actor
   - Executes full operations pipeline (download → frames → faces → headshots)
   - Uses adaptive frame extraction if no headshots found

### Mid-Level Automation (Training + Testing)

2. **run_actor_training.py** - Automated actor training + testing
   - Runs complete training pipeline (steps 1-5)
   - Runs complete testing pipeline (steps 6-10)
   - Iteratively downloads until thresholds met (15+ training, 4+ detected headshots)
   - Copies final model to models directory

3. **run_pipeline_steps.py** - Manual pipeline runner
   - Interactive numbered menu of all 15 pipeline steps
   - Manual step-by-step execution of training pipeline
   - Used for debugging and manual control

### TRAINING PIPELINE (Steps 1-5)

1. **download_actor_images.py** - Downloads from Google Image Search
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

5. **compute_average_embeddings.py** - Creates actor embeddings
   - Generates and averages ArcFace embeddings, saves as .pkl

### TESTING PIPELINE (Steps 6-10)

6. **download_actor_images.py** - Downloads test images (same as step 1)
7. **remove_dupe_training_images.py** - Removes test duplicates (same as step 2)
8. **remove_bad_training_images.py** - Test face validation (keeps 3-10 people for group testing)
9. **eval_star_detection.py** - Detects matching faces in test images
   - Loads reference embeddings, extracts matching face crops

10. **Accept Model** - Copies embedding to models directory

### OPERATIONS PIPELINE (Steps 11-15)

11. **download_video.py** - Downloads videos using yt-dlp
    - Supports YouTube, Vimeo, TikTok, saves to `05_videos/[site]_[id]/`

12. **extract_video_frames.py** - Extracts frames using binary search
    - Saves to `frames/` subfolder with zero-padded naming

13. **extract_frame_faces.py** - Detects faces in frames
    - Creates .pkl files with face data alongside each frame

14. **extract_video_headshots.py** - Extracts actor headshots
    - Matches against reference embeddings, saves top 5 matches

15. **extract_video_thumbnail.py** - Creates video thumbnails
    - Selects frames with most identifiable actors using weighted scoring
    - Generates top 3 thumbnails (thumbnail1.jpg, thumbnail2.jpg, thumbnail3.jpg)
    - Falls back to middle frame if no actors detected

### UTILITY MODULES

16. **utils.py** - Common functions (path conversion, argument parsing, file operations)
17. **utils_deepface.py** - DeepFace utilities with caching
18. **print_pkl.py** - Pickle file inspector
19. **run_integration_test.py** - Integration test script with mock data

## Data Structure

- `00_mocks/` - Mock test data for integration testing
- `01_images/` - Cached Google image search results and manually added images to train on
- `02_training/[actor]/` - Training images + `[actor]_average_embedding.pkl`
- `03_testing/[actor]/` - Test images + `detected_headshots/` subfolder  
- `04_models/[actor]/` - Final accepted models
- `05_videos/[site]_[id]/` - Video file + `frames/` + `headshots/` subfolders
- `05_videos/temp/` - Temporary download directory

## Usage

### Primary Entry Point
```bash
venv/bin/python3 run_headshot_detection.py "https://youtube.com/watch?v=VIDEO_ID" --show "SNL" --actors "Bill Murray, Tina Fey"
```

### Mid-Level Training
```bash
venv/bin/python3 run_actor_training.py "Actor Name" "Show Name"
```

### Manual Control
```bash
venv/bin/python3 run_pipeline_steps.py
```

### Integration Testing
```bash
# Unzip mock data first
unzip mocks.zip

# Run integration test
venv/bin/python3 run_integration_test.py
```

### Manual Commands
```bash
# Training pipeline (steps 1-5)
venv/bin/python3 download_actor_images.py "Name" --training --show "Show" --page 1
venv/bin/python3 remove_dupe_training_images.py --training "Name"
venv/bin/python3 remove_bad_training_images.py --training "Name"
venv/bin/python3 remove_face_outliers.py --training "Name"
venv/bin/python3 compute_average_embeddings.py "Name"

# Testing pipeline (steps 6-10)
venv/bin/python3 download_actor_images.py "Name" --testing --show "Show"
venv/bin/python3 remove_dupe_training_images.py --testing "Name"
venv/bin/python3 remove_bad_training_images.py --testing "Name"
venv/bin/python3 eval_star_detection.py "Name"

# Operations pipeline (steps 11-15)
venv/bin/python3 download_video.py "https://youtube.com/watch?v=VIDEO_ID"
venv/bin/python3 extract_video_frames.py 05_videos/youtube_ABC123/
venv/bin/python3 extract_frame_faces.py 05_videos/youtube_ABC123/
venv/bin/python3 extract_video_headshots.py "Name" 05_videos/youtube_ABC123/
venv/bin/python3 extract_video_thumbnail.py 05_videos/youtube_ABC123/
```

## Dependencies

Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Update dependencies (uses pip-tools):
```bash
pip-compile --upgrade requirements.in
pip install -r requirements.txt
```

Core dependencies are listed in `requirements.in`, with exact versions locked in `requirements.txt`.

**Important**: All scripts must be run with `venv/bin/python3` instead of `python3`. External applications should use the full path: `/path/to/StarMapr/venv/bin/python3 script.py`

## Pipeline Stages

**Training (Steps 1-5)**: Download → Remove Dupes → Remove Bad → Remove Outliers → Generate Embeddings
**Testing (Steps 6-10)**: Download → Remove Dupes → Remove Bad → Detect Faces → Accept Model  
**Operations (Steps 11-15)**: Download Video → Extract Frames → Extract Faces → Extract Headshots → Extract Thumbnails


## Configuration

### Environment Variables (.env)
- **GOOGLE_API_KEY**: your_api_key_here (Google Custom Search API key)
- **GOOGLE_SEARCH_ENGINE_ID**: your_search_engine_id_here (Google Custom Search Engine ID)
- **MAX_DOWNLOAD_PAGES**: 10 (maximum pages to download by google image search)
- **TRAINING_MIN_IMAGES**: 15 (number of good images to find for training)
- **TRAINING_DUPLICATE_THRESHOLD**: 5 (0-64, lower = more strict)
- **TRAINING_OUTLIER_THRESHOLD**: 0.2 (0.0-1.0, lower = more strict)
- **TESTING_DETECTION_THRESHOLD**: 0.4 (0.0-1.0, lower = more strict)
- **TESTING_MIN_HEADSHOTS**: 4 (threshold of headshot detections for successful test)
- **OPERATIONS_EXTRACT_FRAME_COUNT**: 50 (number of frames to extract from videos)
- **OPERATIONS_EXCLUDE_END_SECONDS**: 15 (exclude frames from last N seconds of video)
- **OPERATIONS_HEADSHOT_MATCH_THRESHOLD**: 0.4 (0.0-1.0, lower = more strict)
- **MIN_FACE_SIZE**: 50 (minimum face size for processing, width x height in pixels)

### Key Parameters
- **Model**: ArcFace via DeepFace
- **Similarity**: Cosine similarity
- **Training Face Count**: Exactly 1 (solo portraits)
- **Testing Face Count**: 3-10 (group photos for embedding validation)
- **Headshot Extraction**: Top 5 matches