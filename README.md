# GroupMood

GroupMood is a Python and OpenCV application that analyzes a static group photo. It detects frontal and profile faces, crops detected face regions, applies Haar smile detection, and displays a group summary.

Project documentation: [GroupMood_Documentation.pdf](GroupMood_Documentation.pdf)

## Features

| Feature | Description |
| --- | --- |
| Image operations | Resizing, grayscale conversion, and CLAHE contrast equalization |
| Haar Cascade classifiers | Frontal-face, profile-face, and smile detection |
| Face-ROI analysis | Crops, enlarges, and labels individual detected face regions |
| Summary overlay | Displays total detected faces plus smile-detector and neutral counts |

## Setup

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

## Run

```bash
python groupmood.py path/to/group_photo.jpg
```

Add `--show` to display the result windows. Press any key in a result window to close it.

## Output

Results are saved to `outputs/` by default:

- `01_equalized_grayscale.jpg` — contrast-equalized grayscale image
- `02_face_rois.jpg` — enlarged cropped face regions
- `03_final_emotion_analysis.jpg` — labelled group photo and aggregate statistics

Use `--output my_results` to save to a different folder.

## Limitations

The `Smiling` label means a Haar Cascade found a smile-like feature; it is not a reliable measurement of a person's emotion. Haar Cascades can miss faces or produce false detections, especially with small, side-facing, covered, blurry, or poorly lit faces. This project does not estimate age, gender, or identity.
