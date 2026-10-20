# Fish Freshness Classification

Deep-learning models that classify fish images as **Fresh Fish** or **Non Fresh Fish**.
Three approaches were built and compared:

| # | Approach | Framework | Notebook |
|---|----------|-----------|----------|
| 1 | Custom CNN (Conv2D + BatchNorm + Dropout, L2 regularisation) | TensorFlow / Keras | [`notebooks/01-first-architecture/model3.ipynb`](notebooks/01-first-architecture/model3.ipynb) |
| 2 | Transfer learning with **MobileNetV2** | TensorFlow / Keras | [`notebooks/02-mobilenetv2/`](notebooks/02-mobilenetv2/) |
| 3 | **YOLOv8** classification (Ultralytics, 640px, 100 epochs) | PyTorch / Ultralytics | [`notebooks/03-yolov8/fishfreshness_yolov8.ipynb`](notebooks/03-yolov8/fishfreshness_yolov8.ipynb) |

## Trained models

| File | Description |
|------|-------------|
| `models/mobilenetv2_fish_freshness.h5` | Keras MobileNetV2 model |
| `models/yolov8_best.pt` | YOLOv8 best weights |

## Dataset

The "FISH FRESH AND NON-FRESH" dataset (v1) from Roboflow, with 2 classes. It isn't stored in this repo because it is over GitHub's 100 MB file limit.
The notebooks download it through the Roboflow API:

```python
import os
from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
dataset = rf.workspace("meva-wfywb").project("fish-fresh-and-non-fresh").version(1).download("folder")
```

Set the `ROBOFLOW_API_KEY` environment variable (or Colab secret) to your own Roboflow key.

## Running

The notebooks were written for Google Colab. Open one in Colab, set your Roboflow key, and run all cells.

```bash
pip install tensorflow roboflow ultralytics
```

## Using the trained models

```python
from ultralytics import YOLO
model = YOLO("models/yolov8_best.pt")
print(model("fish.jpg")[0].probs.top1)
```

```python
import tensorflow as tf
model = tf.keras.models.load_model("models/mobilenetv2_fish_freshness.h5")
```
