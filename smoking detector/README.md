# Smoking Detector

A simple computer vision project that detects a "smoking" gesture in a video or webcam stream. It uses hand and face landmarks to measure the distance between a point on the hand and a point on the lips. When the distance falls below a threshold, the frame is labeled **Smoking**, otherwise **No smoking**.

## How it works

1. **Face Mesh** (MediaPipe via cvzone) detects facial landmarks. Landmark `14` is used as the lip point.
2. **Hand Tracking** (MediaPipe via cvzone) detects hand landmarks. Landmark `11` is used as the hand point.
3. The pixel distance between the hand point and the lip point is computed.
4. If `distance < 40`, the label **Smoking** is drawn on the frame. Otherwise, **No smoking**.

## Requirements

- Python 3.8+
- [opencv-python](https://pypi.org/project/opencv-python/)
- [cvzone](https://github.com/cvzone/cvzone)
- [mediapipe](https://pypi.org/project/mediapipe/)

```bash
pip install cvzone mediapipe opencv-python
```

## Usage

### Option 1: Google Colab (video file)

Colab runs on a remote server, so it cannot access your webcam directly. Use a video file stored in Google Drive.

1. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. Set the input path to your video:
   ```python
   input_path = '/content/drive/MyDrive/computer vision/smoking detector/smoking.mp4'
   ```
3. Run the processing cell. The annotated video is written to a temporary file (`/content/temp_output.mp4`), so the original video is never overwritten.

> Files in `/content/` are deleted when the Colab session ends. Download or copy the result to your Drive if you want to keep it.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Hand landmark | `11` | Hand point used for the distance |
| Lip landmark | `14` | Face mesh point used for the distance |
| Distance threshold | `40` | Distance in pixels below which the gesture is labeled "Smoking" |

The threshold is in pixels, so it depends on how far the person is from the camera. Adjust it to fit your video.

## Project structure

```
smoking-detector/
├── smoking_detector.ipynb   # Colab notebook (video file)
└── README.md
```

## Possible improvements

- Normalize the distance by face size to make the threshold independent of camera distance.
- Detect a cigarette object with a trained model (e.g. YOLO) for higher accuracy.
- Require the gesture to persist over several frames before triggering the label.
- Support multiple faces and both hands.

## Credits

- [cvzone](https://github.com/cvzone/cvzone) by Murtaza Hassan
- [MediaPipe](https://developers.google.com/mediapipe) by Google
- [OpenCV](https://opencv.org/)
