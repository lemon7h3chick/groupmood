#!/usr/bin/env python3
"""GroupMood: classical OpenCV analysis of a static group photo.

Topics demonstrated:
  a. Image Operations: resize, grayscale conversion, histogram equalization
  c. Haar Cascade Classifier: frontal-face, profile-face, and smile detection
  d. Image Analysis and Transformation: crop, enlarge, and inspect face ROIs

Smile labels mean that OpenCV's Haar smile detector found a smile-like feature;
they are visual cues, not reliable measurements of a person's emotion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np

Box = tuple[int, int, int, int]


@dataclass
class Detection:
    box: Box
    face_type: str
    status: str = "Neutral"


def load_image(path: Path, max_width: int) -> np.ndarray:
    """Read an image and reduce overly wide input while retaining its ratio."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        image = cv2.resize(image, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)
    return image


def preprocess(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert to grayscale and equalize local contrast for uneven lighting."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalizer = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return gray, equalizer.apply(gray)


def cascade_path(filename: str) -> Path:
    """Find a Haar Cascade in pip OpenCV or common system package locations."""
    data_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    candidates = [Path(data_dir) / filename] if data_dir else []
    candidates.extend(
        [
            Path("/usr/share/opencv5/haarcascades") / filename,
            Path("/usr/share/opencv4/haarcascades") / filename,
            Path("/usr/share/OpenCV/haarcascades") / filename,
        ]
    )
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise RuntimeError(f"Could not find OpenCV Haar Cascade: {filename}")
    return path


def load_cascade(filename: str) -> cv2.CascadeClassifier:
    detector = cv2.CascadeClassifier(str(cascade_path(filename)))
    if detector.empty():
        raise RuntimeError(f"Could not load Haar Cascade: {filename}")
    return detector


def iou(first: Box, second: Box) -> float:
    """Calculate intersection-over-union so duplicate detections can be removed."""
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def unique_boxes(boxes: list[Box], overlap_limit: float = 0.35) -> list[Box]:
    """Keep large boxes first and remove boxes overlapping a kept detection."""
    kept: list[Box] = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        if all(iou(box, existing) < overlap_limit for existing in kept):
            kept.append(box)
    return kept


def detect_faces(enhanced_gray: np.ndarray) -> list[Detection]:
    """Detect frontal faces and both orientations of profile faces."""
    frontal = load_cascade("haarcascade_frontalface_default.xml")
    profile = load_cascade("haarcascade_profileface.xml")
    frontal_boxes = [tuple(map(int, item)) for item in frontal.detectMultiScale(
        enhanced_gray, scaleFactor=1.1, minNeighbors=6, minSize=(100, 100)
    )]
    detections = [Detection(box, "Frontal") for box in unique_boxes(frontal_boxes)]

    profile_boxes = [tuple(map(int, item)) for item in profile.detectMultiScale(
        enhanced_gray, scaleFactor=1.1, minNeighbors=12, minSize=(100, 100)
    )]
    flipped = cv2.flip(enhanced_gray, 1)
    flipped_boxes = profile.detectMultiScale(flipped, scaleFactor=1.1, minNeighbors=12, minSize=(100, 100))
    width = enhanced_gray.shape[1]
    profile_boxes.extend((width - int(x) - int(w), int(y), int(w), int(h)) for x, y, w, h in flipped_boxes)

    for box in unique_boxes(profile_boxes):
        if all(iou(box, detected.box) < 0.35 for detected in detections):
            detections.append(Detection(box, "Profile"))
    return sorted(detections, key=lambda item: (item.box[1], item.box[0]))


def classify_smiles(enhanced_gray: np.ndarray, detections: list[Detection]) -> None:
    """Crop each face ROI and mark it smiling when Haar smile features appear."""
    smile_detector = load_cascade("haarcascade_smile.xml")
    for detected in detections:
        x, y, w, h = detected.box
        face_roi = enhanced_gray[y : y + h, x : x + w]
        lower_face = face_roi[int(h * 0.42) :, :]
        smiles = smile_detector.detectMultiScale(
            lower_face, scaleFactor=1.7, minNeighbors=20,
            minSize=(max(20, w // 4), max(12, h // 8)),
        )
        detected.status = "Smiling" if len(smiles) > 0 else "Neutral"


def make_roi_sheet(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Crop, enlarge, and arrange face ROIs in a labelled contact sheet."""
    tile_width, tile_height, columns = 180, 220, 4
    rows = max(1, (len(detections) + columns - 1) // columns)
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 245, dtype=np.uint8)
    for index, detected in enumerate(detections, start=1):
        x, y, w, h = detected.box
        roi = image[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        scale = min((tile_width - 20) / w, (tile_height - 50) / h)
        resized = cv2.resize(roi, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_CUBIC)
        row, column = divmod(index - 1, columns)
        start_x, start_y = column * tile_width, row * tile_height
        offset_x = start_x + (tile_width - resized.shape[1]) // 2
        offset_y = start_y + 30 + (tile_height - 35 - resized.shape[0]) // 2
        sheet[offset_y : offset_y + resized.shape[0], offset_x : offset_x + resized.shape[1]] = resized
        color = (0, 150, 0) if detected.status == "Smiling" else (0, 100, 220)
        cv2.putText(sheet, f"Face {index}: {detected.status}", (start_x + 8, start_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return sheet


def annotate(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw boxes, per-face expression labels, and aggregate statistics."""
    result = image.copy()
    smiling = sum(item.status == "Smiling" for item in detections)
    neutral = len(detections) - smiling
    for index, detected in enumerate(detections, start=1):
        x, y, w, h = detected.box
        color = (0, 180, 0) if detected.status == "Smiling" else (0, 165, 255)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        label_y = y - 10 if y > 28 else y + h + 20
        cv2.putText(result, f"Face {index}: {detected.status}", (x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    summary = f"Total Faces: {len(detections)} | Smiling: {smiling} | Neutral: {neutral}"
    cv2.rectangle(result, (0, 0), (result.shape[1], 48), (35, 35, 35), -1)
    cv2.putText(result, summary, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def save_results(output_dir: Path, enhanced: np.ndarray, roi_sheet: np.ndarray, result: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "01_equalized_grayscale.jpg"), enhanced)
    cv2.imwrite(str(output_dir / "02_face_rois.jpg"), roi_sheet)
    cv2.imwrite(str(output_dir / "03_final_emotion_analysis.jpg"), result)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze face and smile detections in a static group photo.")
    parser.add_argument("image", type=Path, help="Path to the group-photo image")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Folder for result images")
    parser.add_argument("--max-width", type=int, default=1400, help="Resize very wide input images to this width")
    parser.add_argument("--show", action="store_true", help="Display result windows; press a key to close")
    args = parser.parse_args()
    try:
        image = load_image(args.image, args.max_width)
        _, enhanced = preprocess(image)
        detections = detect_faces(enhanced)
        classify_smiles(enhanced, detections)
        roi_sheet = make_roi_sheet(image, detections)
        result = annotate(image, detections)
        save_results(args.output, enhanced, roi_sheet, result)
    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    smiling = sum(item.status == "Smiling" for item in detections)
    print(f"Total faces: {len(detections)}")
    print(f"Smiling: {smiling} | Neutral: {len(detections) - smiling}")
    print(f"Saved results to: {args.output.resolve()}")
    if args.show:
        cv2.imshow("01 - Equalized grayscale", enhanced)
        cv2.imshow("02 - Cropped face ROIs", roi_sheet)
        cv2.imshow("03 - Group photo emotion analysis", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
