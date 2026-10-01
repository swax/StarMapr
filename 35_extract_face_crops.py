# /// script
# requires-python = ">=3.10"
# dependencies = ["opencv-python>=4.10,<5"]
# ///
"""Extract anonymous, timestamped face-crop candidates from a local video.

Run with `uv run --no-project --script 35_extract_face_crops.py --help`.
This command detects faces only: it does not identify, match, or group people.
"""

import argparse
import json
import math
from pathlib import Path


def sample_times(duration, start, end, interval, explicit=None):
    if not all(math.isfinite(x) for x in (duration, start, interval)):
        raise ValueError("Duration, start, and interval must be finite")
    if duration <= 0 or start < 0 or interval <= 0:
        raise ValueError("Duration and interval must be positive; start must be nonnegative")
    stop = duration if end is None else min(end, duration)
    if not math.isfinite(stop) or stop <= start:
        raise ValueError("End must be finite and after start")
    if explicit is not None:
        if not explicit or any(not math.isfinite(t) or not start <= t < stop for t in explicit):
            raise ValueError("Every timestamp must be within [start, end)")
        return sorted(set(explicit))
    count = math.ceil((stop - start) / interval)
    if count > 10000:
        raise ValueError("More than 10,000 samples requested; increase --interval")
    return [start + i * interval for i in range(count) if start + i * interval < stop]


def crop_bounds(face, width, height, padding):
    x, y, w, h = (int(v) for v in face)
    if w <= 0 or h <= 0 or width <= 0 or height <= 0 or not math.isfinite(padding) or padding < 1:
        raise ValueError("Invalid face dimensions or padding")
    side = min(width, height, math.ceil(max(w, h) * padding))
    # Include forehead/hair above the detector's face rectangle.
    left = max(0, min(width - side, round(x + w / 2 - side / 2)))
    top = max(0, min(height - side, round(y + h * 0.42 - side / 2)))
    return left, top, left + side, top + side


def write_image(cv2, path, image):
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"Could not write {path}")


def contact_sheets(cv2, images, output, prefix, cell_width, cell_height):
    import numpy as np

    paths = []
    for offset in range(0, len(images), 24):
        batch = images[offset:offset + 24]
        rows = math.ceil(len(batch) / 4)
        canvas = np.full((rows * cell_height, 4 * cell_width, 3), 245, dtype=np.uint8)
        for i, (path, label) in enumerate(batch):
            image = cv2.imread(str(path))
            h, w = image.shape[:2]
            scale = min((cell_width - 12) / w, (cell_height - 30) / h)
            tile = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
            top = (i // 4) * cell_height
            left = (i % 4) * cell_width
            canvas[top:top + tile.shape[0], left:left + tile.shape[1]] = tile
            cv2.putText(canvas, label, (left + 3, top + cell_height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        path = output / f"{prefix}-{offset // 24 + 1:02d}.jpg"
        write_image(cv2, path, canvas)
        paths.append(path.name)
    return paths


def extract(args):
    import cv2

    video = args.video.resolve()
    output = args.output.resolve()
    if not video.is_file():
        raise ValueError(f"Video not found: {video}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty to preserve earlier results")
    if args.min_face < 20 or not math.isfinite(args.padding) or args.padding < 1:
        raise ValueError("Minimum face size must be >=20 and padding must be >=1")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError("Video could not be opened")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no usable frame rate")
        times = sample_times(frame_count / fps, args.start, args.end, args.interval, args.timestamps)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if detector.empty():
            raise RuntimeError("OpenCV face detector could not be loaded")
        (output / "frames").mkdir(parents=True, exist_ok=True)
        (output / "crops").mkdir()
        frames, crops, skipped = [], [], []
        for sample_index, seconds in enumerate(times):
            frame_index = round(seconds * fps)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                skipped.append(seconds)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(args.min_face, args.min_face))
            boxes = sorted(boxes, key=lambda b: (int(b[0]), int(b[1])))
            frame_id = f"s{sample_index:04d}-f{frame_index:06d}"
            frame_path = output / "frames" / f"{frame_id}.jpg"
            write_image(cv2, frame_path, frame)
            frames.append({"timestamp_seconds": round(frame_index / fps, 3), "file": frame_path.relative_to(output).as_posix(), "face_count": len(boxes)})
            for number, box in enumerate(boxes):
                x1, y1, x2, y2 = crop_bounds(box, frame.shape[1], frame.shape[0], args.padding)
                crop = frame[y1:y2, x1:x2]
                crop_id = f"{frame_id}-face{number:02d}"
                crop_path = output / "crops" / f"{crop_id}.jpg"
                write_image(cv2, crop_path, crop)
                x, y, w, h = (int(v) for v in box)
                sharpness = cv2.Laplacian(gray[y:y+h, x:x+w], cv2.CV_64F).var()
                crops.append({"id": crop_id, "file": crop_path.relative_to(output).as_posix(), "frame": frames[-1]["file"], "timestamp_seconds": frames[-1]["timestamp_seconds"], "face_box": [x, y, w, h], "crop_box": [x1, y1, x2, y2], "size": [x2-x1, y2-y1], "sharpness": round(float(sharpness), 2)})
        sheets = contact_sheets(cv2, [(output / f["file"], f'{f["timestamp_seconds"]:.2f}s / {f["face_count"]} faces') for f in frames], output, "frames", 320, 210)
        sheets += contact_sheets(cv2, [(output / c["file"], c["id"]) for c in crops], output, "crops", 230, 260)
        result = {"mode": "face_detection_only", "review_required": True, "video": str(video), "fps": fps, "duration_seconds": frame_count / fps, "detector": "opencv-haar-frontalface", "parameters": {"min_face": args.min_face, "padding": args.padding}, "frames": frames, "crops": crops, "skipped_timestamps": skipped, "contact_sheets": sheets}
        (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"manifest": str(output / "manifest.json"), "frames": len(frames), "crops": len(crops), "skipped_frames": len(skipped)}))
        return 0 if frames else 1
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--timestamps", type=float, nargs="+")
    parser.add_argument("--min-face", type=int, default=72)
    parser.add_argument("--padding", type=float, default=1.7)
    args = parser.parse_args()
    try:
        return extract(args)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
