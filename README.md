# Deep Learning for Fish Freshness Detection

Detects whether a fish is **Fresh** or **Non-Fresh** from images of its eye. The eye's clarity, convexity, corneal transparency and colour are reliable freshness indicators.

Final project for the **Higher National Diploma in Data Science** at NIBM – Kandy Innovation Center (2025), by Pramuditha Weerathunga.

## Results

| Model | Type | Accuracy | Precision | Recall | Other |
|-------|------|----------|-----------|--------|-------|
| Baseline CNN (`model3`) | Classification, 224×224 | 84.2% | – | – | Overfits |
| **MobileNetV2 + regularisation (`best_5`)** | Classification, 224×224 | **92.1%** | 91.5% | 92.8% | ROC-AUC ≈ 0.97 |
| **YOLOv8n** | Object detection, 640×640 | ≈87.3% | 88.8% | 90.8% | mAP@0.5 = 96.4% |

- **MobileNetV2 (best_5)** gives the highest accuracy on cropped eye images. It uses an ImageNet-pretrained frozen backbone plus dropout, L2 regularisation, batch normalisation and a Swish dense layer. With a tuned decision threshold of 0.4478, only 12 of 206 spoiled fish were labelled fresh.
- **YOLOv8** finds the eye region in a raw, uncropped photo and classifies it in the same step, which makes it the best choice for a real-time camera or mobile app.

## Repository structure

```
notebooks/
  01-first-architecture/model3.ipynb       Baseline Keras Sequential CNN
  02-mobilenetv2/model_4.ipynb             MobileNetV2 transfer learning
  02-mobilenetv2/best_5.ipynb              Final regularised MobileNetV2 model
  03-yolov8/fishfreshness_yolov8.ipynb     YOLOv8 detection training
models/
  mobilenetv2_fish_freshness.h5            Trained Keras model
  yolov8_best.pt                           Trained YOLOv8 weights
```

## Dataset

"FISH FRESH AND NON-FRESH" (v1) on Roboflow. It contains annotated fish-eye images based on Anas, Jaya & Herdiyeni (2024), [*Measurement and Analysis of Detecting Fish Freshness Levels Using Deep Learning Method*](https://doi.org/10.22146/ijccs.95054).
The dataset isn't stored in this repo because it is over GitHub's 100 MB file limit. The notebooks download it through the Roboflow API:

```python
import os
from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
dataset = rf.workspace("meva-wfywb").project("fish-fresh-and-non-fresh").version(1).download("folder")
```

## Running

The notebooks were written for Google Colab with GPU. Set `ROBOFLOW_API_KEY` to your own Roboflow key, then run all cells.

```bash
pip install tensorflow roboflow ultralytics
```

### Using the trained models

```python
from ultralytics import YOLO
model = YOLO("models/yolov8_best.pt")
results = model("fish.jpg")
results[0].show()
```

```python
import tensorflow as tf
model = tf.keras.models.load_model("models/mobilenetv2_fish_freshness.h5")
```

## Tech stack

Python · TensorFlow / Keras · MobileNetV2 · Ultralytics YOLOv8 · Roboflow · Google Colab
