# Object Detection with SSD MobileNet v3 (OpenCV)

Detects objects in an image using a pre-trained **SSD MobileNet v3 Large** model (trained on the COCO dataset, 80 classes) and OpenCV's DNN module. Each detected object is drawn on the image with a bounding box, its class name and a confidence score.

The notebook is designed to run in **Google Colab**, with the project files stored in Google Drive.

## Project structure

```
Object Detection/
├── object_detection.ipynb                          # Colab notebook
├── coco.names                                      # COCO class names (one per line)
├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt    # Model configuration
├── frozen_inference_graph.pb                       # Pre-trained model weights
├── pexels-eggsy-outis-2151852439-33076681.jpg      # Test image
└── README.md
```

## Requirements

- Python 3
- OpenCV (`opencv-python`)
- Google Colab (or a local Python environment, see below)

## How to run in Google Colab

1. Copy the `Object Detection` folder into your Google Drive under:
   ```
   MyDrive/computer vision/Object Detection/
   ```
2. Open `object_detection.ipynb` in Google Colab.
3. Run the cell. Colab will ask for permission to access your Google Drive; accept it.
4. The image is displayed with all detected objects outlined in red.

If your folder is in a different location, update the `base` variable at the top of the notebook:

```python
base = '/content/drive/MyDrive/computer vision/Object Detection/'
```

## How it works

1. **Load class names** from `coco.names`.
2. **Load and resize the image** to 20% of its original size.
3. **Load the model** with `cv2.dnn_DetectionModel`, using the weights (`.pb`) and config (`.pbtxt`).
4. **Pre-process the input** as expected by the model:
   - input size: 320 × 320
   - scale: 1 / 127.5
   - mean: (127.5, 127.5, 127.5)
   - swap BGR → RGB
5. **Run detection** with a confidence threshold of 0.5.
6. **Draw** a rectangle and a label (`class confidence`) for each detected object.
7. **Display** the result with `cv2_imshow` (Colab's replacement for `cv2.imshow`).

## Customization

| Setting | Where | Effect |
|---|---|---|
| Confidence threshold | `detector.detect(image, confThreshold=0.5)` | Lower it to detect more objects, raise it to keep only confident ones |
| Image size | `cv2.resize(image, (0, 0), None, 0.2, 0.2)` | Change `0.2` to make the displayed image bigger or smaller |
| Test image | `cv2.imread(base + '...jpg')` | Replace with any image in the folder |
| Box / text color | `(0, 0, 255)` | BGR color (red by default) |

## Running locally with a webcam

Colab runs on Google's servers and cannot access your computer's webcam through `cv2.VideoCapture(0)`. To detect objects in real time from your webcam, run the code locally:

```bash
pip install opencv-python
```

Then replace the image loading and display with a capture loop:

```python
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    ids, confidences, boxes = detector.detect(frame, confThreshold=0.5)
    if len(ids) > 0:
        for id, conf, box in zip(ids.flatten(), confidences.flatten(), boxes):
            x, y, w, h = box
            label = Classnames[id - 1] + ' ' + str(round(float(conf), 2))
            cv2.rectangle(frame, box, (0, 0, 255), 2)
            cv2.putText(frame, label, (x, max(y - 10, 20)),
                        cv2.FONT_ITALIC, 0.6, (0, 0, 255), 2)
    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):   # press q to quit
        break
cap.release()
cv2.destroyAllWindows()
```

## Model

- **Architecture:** SSD (Single Shot MultiBox Detector) with a MobileNet v3 Large backbone
- **Dataset:** COCO (80 object classes: person, car, bus, dog, cell phone, etc.)
- **Source:** TensorFlow Object Detection Model Zoo (`ssd_mobilenet_v3_large_coco_2020_01_14`)

## Author

[youssef-bendhaou](https://github.com/youssef-bendhaou)
