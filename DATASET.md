# VisDrone Dataset

## Overview

- **Task:** Object detection (drone-view traffic and pedestrians).
- **Classes (10):** pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor.
- **Splits:** train, val, test (images + labels in YOLO format).

## Layout

```
data/visdrone/
  data.yaml       # paths + 10 class names
  train/
    images/
    labels/
  val/
    images/
    labels/
  test/
    images/
    labels/
```

## Label format (YOLO)

One `.txt` per image; each line: `class_id x_center y_center width height` (normalized 0–1).

## Source

VisDrone: [official site](http://aiskyeye.com/) / [Kaggle](https://www.kaggle.com/datasets/kushagrapandya/visdrone-dataset). Convert annotations to YOLO if needed, then place under `data/visdrone`.
