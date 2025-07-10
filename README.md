# StarMapr

A Python application for actor face recognition and detection using DeepFace with the ArcFace model. This tool was created to complement the [Sketch Comedy Database (SCDB)](https://github.com/swax/SCDB) project by automating the process of scanning comedy sketches for actors and extracting headshots for the [SketchTV](https://www.sketchtv.lol/) website.

## Purpose

StarMapr enables automated identification and extraction of actor faces from video frames or images, making it easier to:
- Identify actors appearing in comedy sketches
- Extract clean headshots for database profiles
- Build comprehensive cast information for sketch comedy shows
- Automate the tedious manual process of actor identification

## Features

- **Actor Image Collection**: Download training images from Google Image Search
- **Data Cleaning**: Remove duplicates and low-quality images automatically
- **Face Consistency Validation**: Remove outlier faces that don't match the target actor
- **Face Embedding Generation**: Create reference embeddings using state-of-the-art ArcFace model
- **Face Detection & Matching**: Identify matching faces in test images with confidence scores
- **Headshot Extraction**: Automatically crop and save detected faces
- **Video Processing**: Download videos and extract frames for face analysis
- **Frame-based Face Detection**: Process video frames to detect and track faces across time

## Installation

1. Clone the repository:
```bash
git clone https://github.com/swax/StarMapr.git
cd StarMapr
```

2. Install Git LFS and pull large files:
```bash
git lfs install
git lfs pull
```

3. Install dependencies:
```bash
pip install deepface numpy opencv-python scikit-learn google-images-search python-dotenv yt-dlp
```

4. Set up configuration:
Create a `.env` file with:
```
# Google Custom Search API Configuration
# Get your API key from: https://developers.google.com/custom-search/v1/introduction
# Get your Search Engine ID from: https://cse.google.com/cse/all
GOOGLE_API_KEY=your_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here

# The max number of pages that can be downloaded by google image search for training/testing purposes
MAX_DOWNLOAD_PAGES=5

# The number of good images to find to do training with
TRAINING_MIN_IMAGES=15

# Training duplicate detection threshold (0-64, lower = more strict)
TRAINING_DUPLICATE_THRESHOLD=5

# Training outlier detection threshold (0.0-1.0, lower = more strict)
TRAINING_OUTLIER_THRESHOLD=0.2

# Testing detection threshold (0.0-1.0, lower = more strict)
TESTING_DETECTION_THRESHOLD=0.4

# The threshold of headshot detections to consider a successful test and the model ready
TESTING_MIN_HEADSHOTS=4

# Operations: number of frames to extract from videos
OPERATIONS_EXTRACT_FRAME_COUNT=50

# Operations: headshot match threshold (0.0-1.0, lower = more strict)
OPERATIONS_HEADSHOT_MATCH_THRESHOLD=0.4

# Minimum face size for processing (width x height in pixels)
MIN_FACE_SIZE=50
```

5. Test the installation:
```bash
# Extract mock data
unzip mocks.zip

# Run integration test
python3 run_integration_test.py
```

## Architecture

StarMapr follows a hierarchical architecture with three execution levels:

```
run_integration_test.py             # Integration test root
└── run_headshot_detection.py       # ★ PRIMARY ENTRY POINT
    ├── run_actor_training.py   # ★ MID-LEVEL automation
    |   |   # TESTING
    │   ├── download_actor_images.py
    │   ├── remove_dupe_training_images.py
    │   ├── remove_bad_training_images.py
    │   ├── remove_face_outliers.py -- testing
    │   ├── compute_average_embeddings.py
    |   |   # TRAINING
    │   ├── download_actor_images.py
    │   ├── remove_dupe_training_images.py
    │   ├── remove_bad_training_images.py
    │   └── eval_star_detection.py
    |   # VIDEO PROCESSING
    ├── download_video.py
    ├── extract_video_frames.py
    ├── extract_frame_faces.py
    └── extract_video_headshots.py
```

## Quick Start

### Primary Entry Point (Recommended)
Complete end-to-end workflow from video URL to extracted headshots:
```bash
python3 run_headshot_detection.py "https://youtube.com/watch?v=VIDEO_ID" --show "SNL" "Bill Murray" "Tina Fey"
python3 run_headshot_detection.py "https://youtube.com/watch?v=VIDEO_ID" --show "SNL" --actors "Bill Murray,Tina Fey,Amy Poehler"
```
**TOP-LEVEL SCRIPT**: This is the main entry point that orchestrates the entire process. It automatically trains actors, downloads video, and extracts headshots.

### Individual Actor Training
For training a single actor without video processing:
```bash
python3 run_actor_training.py "Actor Name" "Show Name"
```
**MID-LEVEL SCRIPT**: Called automatically by `run_headshot_detection.py`, but can be run standalone for actor training.

### Manual Pipeline Control  
For debugging, testing, or manual step-by-step control:
```bash
python3 run_pipeline_steps.py
```
**LOW-LEVEL SCRIPT**: Interactive menu for manual execution of individual pipeline components.

### Manual Pipeline Execution

#### Training Pipeline
```bash
# 1. Download training images (solo portraits)
python3 download_actor_images.py "Bill Murray" --training --show "SNL"

# 2. Remove duplicate images
python3 remove_dupe_training_images.py --training "Bill Murray"

# 3. Remove bad images (keep exactly 1 face)
python3 remove_bad_training_images.py --training "Bill Murray"

# 4. Remove face outliers (detect inconsistent faces)
python3 remove_face_outliers.py --training "Bill Murray"

# 5. Generate reference embeddings
python3 compute_average_embeddings.py "Bill Murray"
```

#### Testing Pipeline
```bash
# 6. Download testing images (group photos)
python3 download_actor_images.py "Bill Murray" --testing --show "SNL"

# 7. Remove duplicate images
python3 remove_dupe_training_images.py --testing "Bill Murray"

# 8. Remove bad images (keep 3-10 faces for group testing)
python3 remove_bad_training_images.py --testing "Bill Murray"

# 9. Detect faces in test images
python3 eval_star_detection.py "Bill Murray"
```

#### Operations Pipeline
```bash
# 1. Download video from supported platforms
python3 download_video.py "https://www.youtube.com/watch?v=-_X904_TZnc"

# 2. Extract representative frames using binary search pattern (script finds video automatically)
python3 extract_video_frames.py videos/youtube_VIDEO_ID/ 50

# 3. Extract face data from all frames (script uses frames/ subfolder automatically)
python3 extract_frame_faces.py videos/youtube_VIDEO_ID/

# 4. Extract actor headshots from video frames
python3 extract_video_headshots.py "Bill Murray" videos/youtube_VIDEO_ID/
```

### Architecture Flow

1. **Integration Test Layer**: `run_integration_test.py` provides end-to-end validation
2. **Application Layer**: `run_headshot_detection.py` orchestrates the complete workflow
3. **Training Layer**: `run_actor_training.py` handles actor model creation
4. **Pipeline Layer**: Individual scripts handle specific data processing tasks

## Project Structure

```
StarMapr/
├── training/                      # Actor training images
│   └── [actor_name]/          # Individual actor folders
├── testing/                       # Test images to process
│   └── detected_headshots/        # Extracted face crops
├── videos/                        # Downloaded videos and extracted frames
│   └── [site]_[video_id]/         # Individual video folders with frames/
├── run_headshot_detection.py      # ★ PRIMARY ENTRY POINT - End-to-end workflow
├── run_actor_training.py      # ★ MID-LEVEL - Actor training automation
├── run_pipeline_steps.py          # ★ LOW-LEVEL - Manual pipeline control
├── run_integration_test.py         # Integration test script with mock data
├── download_actor_images.py   # Google Image Search downloader
├── download_video.py              # Video downloader for multiple platforms
├── extract_video_frames.py        # Video frame extraction using binary search
├── extract_frame_faces.py         # Face detection in video frames
├── extract_video_headshots.py     # Actor headshot extraction from video frames
├── remove_dupe_training_images.py # Duplicate removal tool
├── remove_bad_training_images.py  # Image quality cleaner
├── remove_face_outliers.py        # Face consistency validator
├── compute_average_embeddings.py  # Embedding generator
├── eval_star_detection.py         # Face detection and matching
├── print_pkl.py                   # Pickle file inspection utility
├── utils.py                       # Common utility functions and helpers
└── utils_deepface.py              # DeepFace-specific utilities with caching
```

## Core Components

### Script Hierarchy

#### Top-Level Orchestration (`run_headshot_detection.py`)
- **PRIMARY ENTRY POINT** for end-to-end video processing
- Takes video URL and list of actors as input
- Automatically calls `run_actor_training.py` for each actor
- Downloads video and orchestrates video operations pipeline
- Extracts headshots for all successfully trained actors

#### Mid-Level Automation (`run_actor_training.py`)
- Automated actor training pipeline for individual actors
- Called by `run_headshot_detection.py` but can run standalone
- Iteratively processes images until quality thresholds are met
- Handles both training and testing phases automatically

#### Low-Level Control (`run_pipeline_steps.py`)
- Manual pipeline runner with interactive numbered menu
- Manual step-by-step execution of training pipeline
- Provides numbered menu of all 15 pipeline steps
- Built-in error checking and user-friendly prompts

#### Integration Testing (`run_integration_test.py`)
- Complete end-to-end integration test using mock data
- Tests entire pipeline with hardcoded mock actor and video
- Validates file counts and processing results
- Requires extracting `mocks.zip` to base directory first

### Image Collection (`download_actor_images.py`)
- Downloads actor photos from Google Image Search
- 20 images per page, each page uses different keywords
- Training: different keywords for more face variety
- Testing: keywords targeting group photos
- Automatic folder organization

### Data Cleaning (`remove_dupe_training_images.py`, `remove_bad_training_images.py`, `remove_face_outliers.py`)
- Perceptual hashing for duplicate detection
- Face detection validation
- Face consistency validation using embedding similarity
- Resolution and quality filtering

### Embedding Generation (`compute_average_embeddings.py`)
- Uses DeepFace with ArcFace model
- Computes average embeddings from multiple images
- Saves reference embeddings as pickle files

### Face Detection (`eval_star_detection.py`)
- Loads precomputed reference embeddings
- Processes test images for matching faces
- Extracts and saves face crops with similarity scores
- Configurable similarity thresholds

### Video Processing (`download_video.py`, `extract_video_frames.py`, `extract_frame_faces.py`, `extract_video_headshots.py`)
- Downloads videos from YouTube, Vimeo, TikTok, and other platforms using yt-dlp
- Extracts representative frames using binary search pattern for optimal coverage
- Detects faces in extracted frames with bounding boxes and embeddings
- Saves face metadata for each frame to enable temporal analysis
- Extracts actor headshots from video frames using similarity matching

## Configuration

All default values are configurable through environment variables in the `.env` file:

- **Google API credentials**: Required for image downloading (`GOOGLE_API_KEY`, `GOOGLE_SEARCH_ENGINE_ID`)
- **Maximum download pages**: 5 pages (`MAX_DOWNLOAD_PAGES`)
- **Training minimum images**: 15 images (`TRAINING_MIN_IMAGES`)
- **Training duplicate threshold**: 5 Hamming distance, 0-64 scale (`TRAINING_DUPLICATE_THRESHOLD`)
- **Training outlier threshold**: 0.2 cosine similarity, 0.0-1.0 scale (`TRAINING_OUTLIER_THRESHOLD`)
- **Testing detection threshold**: 0.4 cosine similarity, 0.0-1.0 scale (`TESTING_DETECTION_THRESHOLD`)
- **Testing minimum headshots**: 4 detected headshots (`TESTING_MIN_HEADSHOTS`)
- **Frame extraction count**: 50 frames (`OPERATIONS_EXTRACT_FRAME_COUNT`)
- **Headshot match threshold**: 0.4 cosine similarity, 0.0-1.0 scale (`OPERATIONS_HEADSHOT_MATCH_THRESHOLD`)
- **Minimum face size**: 50 pixels (`MIN_FACE_SIZE`)

All thresholds are adjustable with command-line `--threshold` flags.

**Technical specs**:
- **Supported formats**: .gif, .jpg, .jpeg, .png, .bmp, .tiff, .webp
- **Face detection model**: ArcFace via DeepFace
- **Similarity metric**: Cosine similarity

## Integration with SCDB

StarMapr was designed to streamline actor identification for the Sketch Comedy Database:

1. **Download videos** of comedy sketches from various platforms
2. **Extract representative frames** using optimized sampling techniques
3. **Process frames** through StarMapr to identify known actors
4. **Extract headshots** automatically for database profiles
5. **Build cast lists** with confidence scores and temporal data
6. **Populate SCDB** with identified actors and clean headshot images

Visit [SketchTV.lol](https://www.sketchtv.lol/) to see the results in action!

## Troubleshooting

### Video Headshot Extraction Issues

If the correct headshots are not being found for a video, follow these steps to improve accuracy:

1. **Check Training Data Quality**
   - Ensure outliers have been pruned effectively from training images
   - Verify that remaining training images are actually of the target actor
   - Use `remove_face_outliers.py` with adjusted threshold if needed:
     ```bash
     python3 remove_face_outliers.py --training "Actor Name" --threshold 0.05
     ```

2. **Adjust Outlier Detection Threshold**
   - Lower threshold (e.g., 0.05) = stricter outlier removal
   - Higher threshold (e.g., 0.2) = more lenient outlier removal
   - Edit `.env` file: `TRAINING_OUTLIER_THRESHOLD=0.05`

3. **Regenerate Average Embeddings**
   - After cleaning training data, regenerate the reference embeddings:
     ```bash
     python3 compute_average_embeddings.py "Actor Name"
     ```

4. **Test Detection Accuracy**
   - Run testing pipeline with new average embeddings to verify improved accuracy:
     ```bash
     python3 eval_star_detection.py "Actor Name"
     ```

5. **Retry Video Headshot Extraction**
   - Extract headshots from video using the improved reference embeddings:
     ```bash
     python3 extract_video_headshots.py "Actor Name" videos/youtube_VIDEO_ID/
     ```

6. **Increase Training/Testing Data**
   - If insufficient data was found, download additional pages:
     ```bash
     python3 download_actor_images.py "Actor Name" --training --show "Show Name" --page 2
     python3 download_actor_images.py "Actor Name" --testing --show "Show Name" --page 3
     ```
   - Each page downloads 20 more images using different keywords for variety

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE.md).