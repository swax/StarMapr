# StarMapr

A Python application for celebrity face recognition and detection using DeepFace with the ArcFace model. This tool was created to complement the [Sketch Comedy Database (SCDB)](https://github.com/swax/SCDB) project by automating the process of scanning comedy sketches for actors and extracting headshots for the [SketchTV](https://www.sketchtv.lol/) website.

## Purpose

StarMapr enables automated identification and extraction of celebrity faces from video frames or images, making it easier to:
- Identify actors appearing in comedy sketches
- Extract clean headshots for database profiles
- Build comprehensive cast information for sketch comedy shows
- Automate the tedious manual process of actor identification

## Features

- **Celebrity Image Collection**: Download training images from Google Image Search
- **Data Cleaning**: Remove duplicates and low-quality images automatically
- **Face Consistency Validation**: Remove outlier faces that don't match the target celebrity
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

2. Install dependencies:
```bash
pip install deepface numpy opencv-python scikit-learn google-images-search python-dotenv yt-dlp
```

3. Set up Google API credentials (for image downloading):
Create a `.env` file with:
```
GOOGLE_API_KEY=your_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id_here
```

## Quick Start

### Interactive Pipeline Runner (Recommended)
Run the complete pipeline interactively with guided menu options:
```bash
python3 run_pipeline.py
```
This script provides a numbered menu to execute each pipeline step in order, with automatic path management and validation.

### Manual Pipeline Execution

#### Training Pipeline
```bash
# 1. Download training images (solo portraits)
python3 download_celebrity_images.py "Bill Murray" 20 --training

# 2. Remove duplicate images
python3 remove_dupe_training_images.py --training "Bill Murray"

# 3. Remove bad images (keep exactly 1 face)
python3 remove_bad_training_images.py --training "Bill Murray"

# 4. Remove face outliers (detect inconsistent faces)
python3 remove_face_outliers.py --training "Bill Murray"

# 5. Generate reference embeddings
python3 compute_average_embeddings.py training/bill_murray/
```

#### Testing Pipeline
```bash
# 6. Download testing images (group photos)
python3 download_celebrity_images.py "Bill Murray" 15 --testing

# 7. Remove duplicate images
python3 remove_dupe_training_images.py --testing "Bill Murray"

# 8. Remove bad images (keep 4-10 faces)
python3 remove_bad_training_images.py --testing "Bill Murray"

# 9. Detect faces in test images
python3 eval_star_detection.py "Bill Murray"
```

#### Video Processing Pipeline
```bash
# 1. Download video from supported platforms
python3 download_video.py "https://www.youtube.com/watch?v=VIDEO_ID"

# 2. Extract representative frames using binary search pattern
python3 extract_video_frames.py videos/youtube_VIDEO_ID/video_file.mp4 50

# 3. Extract face data from all frames
python3 extract_frame_faces.py videos/youtube_VIDEO_ID/frames/
```

## Project Structure

```
StarMapr/
├── training/                    # Celebrity training images
│   └── [celebrity_name]/        # Individual celebrity folders
├── testing/                     # Test images to process
│   └── detected_headshots/      # Extracted face crops
├── videos/                      # Downloaded videos and extracted frames
│   └── [site]_[video_id]/       # Individual video folders with frames/
├── run_pipeline.py              # Interactive pipeline runner
├── download_celebrity_images.py # Google Image Search downloader
├── download_video.py            # Video downloader for multiple platforms
├── extract_video_frames.py      # Video frame extraction using binary search
├── extract_frame_faces.py       # Face detection in video frames
├── remove_dupe_training_images.py # Duplicate removal tool
├── remove_bad_training_images.py # Image quality cleaner
├── remove_face_outliers.py      # Face consistency validator
├── compute_average_embeddings.py # Embedding generator
└── eval_star_detection.py      # Face detection and matching
```

## Core Components

### Pipeline Runner (`run_pipeline.py`)
- Interactive menu-driven pipeline execution
- Automatic path management and validation
- Step-by-step guidance through training and testing workflows
- Built-in error checking and user-friendly prompts

### Image Collection (`download_celebrity_images.py`)
- Downloads celebrity photos from Google Image Search
- Optimized search parameters for face portraits
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

### Video Processing (`download_video.py`, `extract_video_frames.py`, `extract_frame_faces.py`)
- Downloads videos from YouTube, Vimeo, TikTok, and other platforms using yt-dlp
- Extracts representative frames using binary search pattern for optimal coverage
- Detects faces in extracted frames with bounding boxes and embeddings
- Saves face metadata for each frame to enable temporal analysis

## Configuration

- **Default similarity threshold**: 0.6 (adjustable with `--threshold`)
- **Supported formats**: .jpg, .jpeg, .png, .bmp, .tiff, .webp
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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE.md).