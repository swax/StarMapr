# StarMapr

A Python application for actor face recognition and detection using DeepFace with the ArcFace model. This tool was created to complement the [Sketch Comedy Database (SCDB)](https://github.com/swax/SCDB) project by automating the process of scanning comedy sketches for actors and extracting headshots for the [SketchTV](https://www.sketchtv.lol/) website.

## Purpose

StarMapr enables automated identification and extraction of actor faces from video frames or images, making it easier to:
- Identify actors appearing in comedy sketches
- Extract clean headshots for database profiles
- Build comprehensive cast information for sketch comedy shows
- Automate the tedious manual process of actor identification

## Features

See [automated validation and AWS setup](AUTOMATED_VALIDATION.md) for the quality
gates, explicit no-headshot outcomes, model migration, and held-out benchmark.

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

3. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

4. Install dependencies:
```bash
uv sync
```

To update dependencies to their latest versions:
```bash
uv lock --upgrade
uv sync
```

5. Set up configuration:
Create a `.env` file with:
```
# Google Custom Search API Configuration
# Get your API key from: https://developers.google.com/custom-search/v1/introduction
# Get your Search Engine ID from: https://cse.google.com/cse/all
GOOGLE_API_KEY=your_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here

# The max number of pages that can be downloaded by google image search for training/testing purposes
MAX_DOWNLOAD_PAGES=10

# The number of good images to find to do training with
TRAINING_MIN_IMAGES=15

# Training duplicate detection threshold (0-64, lower = more strict)
TRAINING_DUPLICATE_THRESHOLD=5

# Training outlier detection threshold (0.0-1.0, lower = more strict)
TRAINING_OUTLIER_THRESHOLD=0.2

# Faces from the page 1 "{actor} {show}" search anchor identity; later faces must match it (higher = more strict)
TRAINING_ANCHOR_THRESHOLD=0.4

# Testing detection threshold (0.0-1.0, lower = more strict)
TESTING_DETECTION_THRESHOLD=0.4

# The threshold of headshot detections to consider a successful test and the model ready
TESTING_MIN_HEADSHOTS=4

# Ignore training and test faces this similar to a blank image's embedding (tiny, blurred or drawn faces; lower = more strict)
MAX_BLANK_SIMILARITY=0.5

# Operations: number of frames to extract from videos
OPERATIONS_EXTRACT_FRAME_COUNT=50

# Operations: headshot match threshold (0.0-1.0, lower = more strict)
OPERATIONS_HEADSHOT_MATCH_THRESHOLD=0.4

# Minimum face size for processing (width x height in pixels)
MIN_FACE_SIZE=50
```

6. Test the installation:
```bash
# Extract mock data
unzip 00_mocks.zip

# Run integration test
uv run python 00_run_integration_test.py # On Windows: uv run python 00_run_integration_test.py 
```

**Note**: All scripts should be run using `uv run python` instead of `python3`. External applications should run: `cd /path/to/StarMapr && uv run python script.py`

## Architecture

StarMapr follows a hierarchical architecture with four execution levels:

```
00_run_integration_test.py                 # Integration test root
└── 01_run_headshot_detection.py           # ★ PRIMARY ENTRY POINT
    ├── 02_run_actor_training.py       # ★ MID-LEVEL orchestration
    │   ├── 03_run_training_pipeline.py    # Training automation
    │   │   ├── 10_download_actor_images.py
    │   │   ├── 11_remove_dupe_training_images.py
    │   │   ├── 12_remove_bad_training_images.py
    │   │   ├── 13_remove_face_outliers.py
    │   │   ├── 14_cluster_and_keep_largest.py
    │   │   └── 15_compute_average_embeddings.py
    │   └── 04_run_testing_pipeline.py     # Testing automation
    │       ├── 10_download_actor_images.py
    │       ├── 11_remove_dupe_training_images.py
    │       ├── 12_remove_bad_training_images.py
    │       └── 20_eval_star_detection.py
    ├── 30_download_video.py               # VIDEO PROCESSING
    ├── 31_extract_video_frames.py
    ├── 32_extract_frame_faces.py
    ├── 33_extract_video_headshots.py
    └── 34_extract_video_thumbnail.py
```

## Quick Start

### Primary Entry Point (Recommended)
Complete end-to-end workflow from video URL to extracted headshots:
```bash
uv run python 01_run_headshot_detection.py "https://youtube.com/watch?v=VIDEO_ID" --show "SNL" "Bill Murray" "Tina Fey"
uv run python 01_run_headshot_detection.py "https://youtube.com/watch?v=VIDEO_ID" --show "SNL" --actors "Bill Murray,Tina Fey,Amy Poehler"
```
**TOP-LEVEL SCRIPT**: This is the main entry point that orchestrates the entire process. It automatically trains actors, downloads video, and extracts headshots.

### Complete Actor Training (Training + Testing)
For training and testing a single actor without video processing:
```bash
uv run python 02_run_actor_training.py "Actor Name" "Show Name"
```
**MID-LEVEL ORCHESTRATION**: Called automatically by `01_run_headshot_detection.py`, but can be run standalone. Orchestrates both training and testing pipelines.

### Failed training cooldown and source domains

An unsuccessful training, testing, or model-promotion run starts a **seven-day
cooldown for that actor**, across shows. Subsequent calls skip the entire training
and testing workflow before archiving folders or issuing searches. Skips do not
extend the deadline. Valid models are still reused immediately. Quality abstention
also starts a cooldown, and optional headshot processing can continue when a call
is skipped. Both `--retrain` and automatic stale-model retraining respect cooldowns;
use `--ignore-cooldown` explicitly to retry early after fixing the cause.

StarMapr records failed runs that start cooldowns and attempts skipped during them,
both in total and per actor. The durable local database is
`07_training_stats/training.sqlite`, separate from archived training folders. Counts
start when this feature is installed; historical logs are not imported. A skipped
attempt is not an API-query or dollar-savings estimate, because some retries would
have reused cached images. Back up this folder with your local models and caches.

For newly downloaded images, `image-sources.json` preserves the image URL, source
page URL, query, and file hash in both the cache and working folder, including when
copies get new filenames. After a model passes training, testing, and promotion,
its contributing images are recorded in model metadata and the statistics database.
The domain report ranks **source-page domains** by unique accepted image hashes,
with actor and successful-run counts; repeated retraining does not inflate the
unique image count. Image-host domains (often CDNs) are tracked separately in JSON.
This is an evidence-based starting list for future domain-restricted search.

Old cached or manual images without provenance remain usable and are counted as
unknown sources. Rejected images and unsuccessful models do not add domains.
No searches are made to reconstruct missing provenance. Statistics are local to
this checkout and ignored by Git; they are not shared between computers.

```bash
uv run python 93_training_stats.py
uv run python 93_training_stats.py --json
uv run python 93_training_stats.py --domains-csv 07_training_stats/source-domains.csv
```

### Individual Pipeline Scripts
For running only the training or testing phase independently:
```bash
# Training pipeline only (creates embeddings)
uv run python 03_run_training_pipeline.py "Actor Name" "Show Name"

# Testing pipeline only (validates model with detection tests)
uv run python 04_run_testing_pipeline.py "Actor Name" "Show Name"
```
**PIPELINE SCRIPTS**: Run specific phases independently. Training must complete before testing can run.
These low-level manual scripts bypass the automatic cooldown; use
`02_run_actor_training.py` for cooldown enforcement and successful-run statistics.

### Manual Pipeline Control
For debugging, testing, or manual step-by-step control:
```bash
uv run python 05_run_pipeline_steps.py
```
**LOW-LEVEL SCRIPT**: Interactive menu for manual execution of individual pipeline components.

### Manual Pipeline Execution

#### Training Pipeline
```bash
# 1. Download training images (solo portraits)
uv run python 10_download_actor_images.py "Bill Murray" --training --show "SNL"

# 2. Remove duplicate images
uv run python 11_remove_dupe_training_images.py --training "Bill Murray"

# 3. Remove bad images (keep exactly 1 face)
uv run python 12_remove_bad_training_images.py --training "Bill Murray"

# 4. Remove face outliers (detect inconsistent faces)
uv run python 13_remove_face_outliers.py --training "Bill Murray"

# 5. Generate reference embeddings
uv run python 15_compute_average_embeddings.py "Bill Murray"
```

#### Testing Pipeline
```bash
# 6. Download testing images (group photos)
uv run python 10_download_actor_images.py "Bill Murray" --testing --show "SNL"

# 7. Remove duplicate images
uv run python 11_remove_dupe_training_images.py --testing "Bill Murray"

# 8. Remove bad images (keep 3-10 faces for group testing)
uv run python 12_remove_bad_training_images.py --testing "Bill Murray"

# 9. Detect faces in test images
uv run python 20_eval_star_detection.py "Bill Murray"
```

#### Operations Pipeline
```bash
# 1. Download video from supported platforms
uv run python 30_download_video.py "https://www.youtube.com/watch?v=-_X904_TZnc"

# 2. Extract representative frames using binary search pattern (script finds video automatically)
uv run python 31_extract_video_frames.py videos/youtube_VIDEO_ID/ 50

# 3. Extract face data from all frames (script uses frames/ subfolder automatically)
uv run python 32_extract_frame_faces.py videos/youtube_VIDEO_ID/

# 4. Extract actor headshots from video frames
uv run python 33_extract_video_headshots.py "Bill Murray" videos/youtube_VIDEO_ID/

# 5. Create video thumbnails (selects frames with most actors)
uv run python 34_extract_video_thumbnail.py videos/youtube_VIDEO_ID/
```

### Architecture Flow

1. **Integration Test Layer**: `00_run_integration_test.py` provides end-to-end validation
2. **Application Layer**: `01_run_headshot_detection.py` orchestrates the complete workflow
3. **Orchestration Layer**: `02_run_actor_training.py` coordinates training and testing phases
4. **Pipeline Layer**: `03_run_training_pipeline.py` and `04_run_testing_pipeline.py` handle specific phases
5. **Component Layer**: Individual scripts handle specific data processing tasks

## Project Structure

```
StarMapr/
├── training/                         # Actor training images
│   └── [actor_name]/                 # Individual actor folders
├── testing/                          # Test images to process
│   └── detected_headshots/           # Extracted face crops
├── videos/                           # Downloaded videos and extracted frames
│   └── [site]_[video_id]/            # Individual video folders with frames/
├── 00_run_integration_test.py        # Integration test script with mock data
├── 01_run_headshot_detection.py      # ★ PRIMARY ENTRY POINT - End-to-end workflow
├── 02_run_actor_training.py          # ★ MID-LEVEL - Actor training/testing orchestration
├── 03_run_training_pipeline.py       # Training pipeline automation
├── 04_run_testing_pipeline.py        # Testing pipeline automation
├── 05_run_pipeline_steps.py          # ★ LOW-LEVEL - Manual pipeline control
├── 10_download_actor_images.py       # Google Image Search downloader
├── 11_remove_dupe_training_images.py # Duplicate removal tool
├── 12_remove_bad_training_images.py  # Image quality cleaner
├── 13_remove_face_outliers.py        # Face consistency validator
├── 14_cluster_and_keep_largest.py    # Clustering-based outlier detection
├── 15_compute_average_embeddings.py  # Embedding generator
├── 20_eval_star_detection.py         # Face detection and matching
├── 30_download_video.py              # Video downloader for multiple platforms
├── 31_extract_video_frames.py        # Video frame extraction using binary search
├── 32_extract_frame_faces.py         # Face detection in video frames
├── 33_extract_video_headshots.py     # Actor headshot extraction from video frames
├── 34_extract_video_thumbnail.py     # Video thumbnail creation from best frames
├── 92_print_pkl.py                   # Pickle file inspection utility
├── utils.py                          # Common utility functions and helpers
└── utils_deepface.py                 # DeepFace-specific utilities with caching
```

## Core Components

### Script Hierarchy

#### Top-Level Orchestration (`01_run_headshot_detection.py`)
- **PRIMARY ENTRY POINT** for end-to-end video processing
- Takes video URL and list of actors as input
- Automatically calls `02_run_actor_training.py` for each actor
- Downloads video and orchestrates video operations pipeline
- Extracts headshots for all successfully trained actors

#### Mid-Level Orchestration (`02_run_actor_training.py`)
- Orchestrates complete actor training and testing workflow
- Called by `01_run_headshot_detection.py` but can run standalone
- Calls `03_run_training_pipeline.py` and `04_run_testing_pipeline.py` in sequence
- Handles model existence checks and folder cleanup
- Copies successful models to models directory

#### Pipeline Automation Scripts

##### Training Pipeline (`03_run_training_pipeline.py`)
- Automated training pipeline for individual actors
- Iteratively downloads and processes training images
- Removes duplicates, bad images, and outliers
- Uses both similarity-based and clustering-based outlier detection
- Generates average embeddings for the actor
- Can be run independently of testing phase

##### Testing Pipeline (`04_run_testing_pipeline.py`)
- Automated testing pipeline for model validation
- Iteratively downloads and processes testing images (group photos)
- Runs face detection to validate model accuracy
- Requires training embeddings to exist first
- Can be run independently after training completes

#### Low-Level Control (`05_run_pipeline_steps.py`)
- Manual pipeline runner with interactive numbered menu
- Manual step-by-step execution of individual components
- Provides numbered menu of all 15 pipeline steps
- Built-in error checking and user-friendly prompts

#### Integration Testing (`00_run_integration_test.py`)
- Complete end-to-end integration test using mock data
- Tests entire pipeline with hardcoded mock actor and video
- Validates file counts and processing results
- Requires extracting `mocks.zip` to base directory first

### Image Collection (`10_download_actor_images.py`)
- Downloads actor photos from Google Image Search
- 20 images per page, each page uses different keywords
- Training: different keywords for more face variety; large, face-dominant results
- Testing: keywords targeting group photos; pages 3-4 add the show name so namesakes' photos don't crowd out the cast
- Automatic folder organization
- Never reuses StarMapr's own video headshots (`*_match_*_position_*`) as training images

### Data Cleaning (`11_remove_dupe_training_images.py`, `12_remove_bad_training_images.py`, `13_remove_face_outliers.py`)
- Perceptual hashing for duplicate detection
- Face detection validation
- Sets aside training faces close to a blank image's embedding (blurry, tiny, drawn, or detector false positives) in `low_information/`
- Anchors identity on the page 1 "{actor} {show}" search, setting aside namesakes and co-stars in `off_anchor/`
- Face consistency validation using embedding similarity
- Resolution and quality filtering

### Embedding Generation (`15_compute_average_embeddings.py`)
- Uses DeepFace with ArcFace model
- Computes average embeddings from multiple images
- Saves reference embeddings as pickle files

### Face Detection (`20_eval_star_detection.py`)
- Loads precomputed reference embeddings
- Processes test images for matching faces
- Counts at most one face per image: the best match, which must beat other actors' models by the competitor margin
- Ignores low-information faces (tiny, blurred or drawn) whose embedding is close to a blank image's
- Extracts and saves face crops with similarity scores
- Configurable similarity thresholds

### Video Processing (`30_download_video.py`, `31_extract_video_frames.py`, `32_extract_frame_faces.py`, `33_extract_video_headshots.py`, `34_extract_video_thumbnail.py`)
- Downloads videos from YouTube, Vimeo, TikTok, and other platforms using yt-dlp
- Extracts representative frames using binary search pattern for optimal coverage
- Detects faces in extracted frames with bounding boxes and embeddings
- Saves face metadata for each frame to enable temporal analysis
- Extracts actor headshots from video frames using similarity matching
- Creates video thumbnails by selecting frames with most identifiable actors using weighted scoring

## Configuration

All default values are configurable through environment variables in the `.env` file:

- **Google API credentials**: Required for image downloading (`GOOGLE_API_KEY`, `GOOGLE_SEARCH_ENGINE_ID`)
- **Maximum download pages**: 10 pages (`MAX_DOWNLOAD_PAGES`)
- **Training minimum images**: 15 images (`TRAINING_MIN_IMAGES`)
- **Training duplicate threshold**: 5 Hamming distance, 0-64 scale (`TRAINING_DUPLICATE_THRESHOLD`)
- **Training outlier threshold**: 0.2 cosine similarity, 0.0-1.0 scale (`TRAINING_OUTLIER_THRESHOLD`)
- **Testing detection threshold**: 0.4 cosine similarity, 0.0-1.0 scale (`TESTING_DETECTION_THRESHOLD`)
- **Testing minimum headshots**: 4 detected headshots (`TESTING_MIN_HEADSHOTS`)
- **Blank-image similarity limit**: 0.5 cosine similarity, 0.0-1.0 scale (`MAX_BLANK_SIMILARITY`)
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
   - Use `13_remove_face_outliers.py` with adjusted threshold if needed:
     ```bash
     uv run python 13_remove_face_outliers.py --training "Actor Name" --threshold 0.05
     ```

2. **Adjust Outlier Detection Threshold**
   - Lower threshold (e.g., 0.05) = stricter outlier removal
   - Higher threshold (e.g., 0.2) = more lenient outlier removal
   - Edit `.env` file: `TRAINING_OUTLIER_THRESHOLD=0.05`

3. **Regenerate Average Embeddings**
   - After cleaning training data, regenerate the reference embeddings:
     ```bash
     uv run python 15_compute_average_embeddings.py "Actor Name"
     ```

4. **Test Detection Accuracy**
   - Run testing pipeline with new average embeddings to verify improved accuracy:
     ```bash
     uv run python 20_eval_star_detection.py "Actor Name"
     ```

5. **Retry Video Headshot Extraction**
   - Extract headshots from video using the improved reference embeddings:
     ```bash
     uv run python 33_extract_video_headshots.py "Actor Name" videos/youtube_VIDEO_ID/
     ```

6. **Increase Training/Testing Data**
   - If insufficient data was found, download additional pages:
     ```bash
     uv run python 10_download_actor_images.py "Actor Name" --training --show "Show Name" --page 2
     uv run python 10_download_actor_images.py "Actor Name" --testing --show "Show Name" --page 3
     ```
   - Each page downloads 20 more images using different keywords for variety

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE.md).

## Face detection and cropping without identification

To prepare character images from a local video without loading actor models, use
the standalone crop command. Its inline dependency metadata installs only OpenCV 4
and its numerical dependencies; a full `uv sync`, credentials, training images, and
cloud services are unnecessary.

```powershell
uv run --no-project --script 35_extract_face_crops.py "C:\media\sketch.mp4" "C:\media\sketch-crops" --start 3 --end 264 --interval 8
```

Use `--timestamps 51 123 131 251` instead of interval sampling to revisit specific
moments. The output directory must be empty. Results include original frames,
JPEG crops using the standard StarMapr headshot framing at native resolution, contact sheets, and `manifest.json`
with timestamps, pixel bounds, and sharpness measurements. The shared crop helper
adds 1.5 face-widths on each side, 0.5 face-heights above, and 1.5 below; crops
that hit an image edge are rejected, as in the existing extraction stage. The
previous tight square layout is available explicitly with `--crop-style square`.
No identities,
embeddings, or cross-frame face matching are produced.

These are candidates for manual review: the detector can miss profile faces or
mistake background objects for faces. Select clear, unobstructed crops and label
fictional characters from dialogue/scene context and independently verified cast
credits. Sharpness is a focus measurement, not an identity or acceptance score.
No detections produces an empty crop list; nothing is automatically published.

Run the crop geometry and video-decoding checks with:

```powershell
uv run --no-project --with "opencv-python>=4.10,<5" python -m unittest discover -s tests -p test_face_crops.py -v
```
