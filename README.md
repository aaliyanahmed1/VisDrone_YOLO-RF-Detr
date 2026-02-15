# VisDrone Object Detection – YOLO26 & RF-DETR

A practical implementation of object detection on **VisDrone** (drone-captured traffic and pedestrian scenes) using two models: an efficient **CNN baseline (YOLO26)** and an advanced **transformer-based detector (RF-DETR)**. This project covers the complete pipeline from data preparation and training to evaluation and deployment, including an inference API with a web UI. This documentation guides through the entire process and contains all the steps taken to develop and evaluate both models.

## Contents

- [Technical Architecture](#technical-architecture)
- [Problem & Dataset](#problem--dataset-choice)
- [Data Understanding & Preparation](#data-understanding--preparation)
- [Model Selection & Training](#model-selection--training)
- [Evaluation & Error Analysis](#evaluation--error-analysis)
- [Model Performance & Test Results](#model-performance--test-results-visdrone2019-det-test-dev)
- [Inference & Deployment](#inference--deployment)
- [File Structure](#file-structure)
- [Quick Start](#quick-start)
- [Summary](#summary)

---

## Technical Architecture

The project employs two detector families: a **CNN-based** single-stage detector (YOLO26) as an efficient baseline and a **transformer-based** detector (RF-DETR) as an advanced model.

### Model Structure Overview

#### YOLO26 (CNN)
- **Architecture Type**: CNN (Convolutional Neural Network), single-stage detector
- **Role**: Efficient baseline — fast to train and run, good accuracy–compute tradeoff
- **Design**: Convolutional backbone, neck, and detection head; predicts bounding boxes and class scores in one forward pass
- **Input**: Images resized to fixed size (e.g. 640×640); normalization applied in the pipeline
- **Training**: Pretrained on COCO; fine-tuned on VisDrone; hyperparameters set in the training script and tunable via CLI
- **Output**: Bounding boxes (xyxy or xywh), class IDs, and confidence scores for the 10 VisDrone classes

#### RF-DETR (Transformer)
- **Architecture Type**: Transformer-based (DETR-style with refinements)
- **Role**: Advanced model — higher potential accuracy; more compute than the CNN baseline
- **Design**: Encoder–decoder transformer; encoder processes image features; decoder attends to object queries and produces box and class predictions; no hand-designed NMS; end-to-end set prediction
- **Input**: COCO-format annotations and images; preprocessing and augmentation as provided by the library
- **Training**: Pretrained weights; fine-tuned on VisDrone (COCO format); configurable epochs, batch size, and learning rate
- **Output**: Bounding boxes and class scores (COCO-style); mapped to the same 10 VisDrone classes for evaluation and inference

**Transfer Learning**: Both models use pretrained weights and are fine-tuned on VisDrone. This approach is justified by limited data and compute; training from scratch would be slower and less effective.

---

## Problem & Dataset Choice

**Task:** Object detection — predict bounding boxes and class labels for each object in an image.

**Dataset: VisDrone**
- **Description**: Public benchmark of images and videos from drone viewpoints (traffic, pedestrians, vehicles)
- **Rationale**: Non-trivial (10 classes, small and distant objects, real-world clutter); widely used for detection; suitable for limited-compute training
- **Label Format**: YOLO format (normalized `class_id x_center y_center width height` per line). Conversion to COCO format is used for RF-DETR where required
- **Classes (10)**: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor

---

## Data Understanding & Preparation

- **Dataset Size**: Depends on the VisDrone split (train/val/test). The check and analyze scripts report exact counts
- **Inspection**:
  - `python scripts/check_dataset.py --data-root <dataset_root>` — validates structure, missing files, invalid labels
  - `python scripts/analyze_data.py --data <dataset_root>` — per-split image and box counts, and class distribution
- **Preprocessing**: Resize and normalization are handled inside the YOLO and RF-DETR training pipelines (e.g. 640px, ImageNet-style normalization)
- **Augmentation**: YOLO uses built-in augmentation (flip, mosaic, etc.); RF-DETR uses its library augmentation when available
- **Class Imbalance**: VisDrone is imbalanced (e.g. many more cars than awning-tricycles). The pipeline uses default loss and augmentation; class weights or oversampling can be added if needed
- **Noisy Labels**: The check script reports invalid or missing labels; no explicit noise cleaning is applied

**Expected Layout**: Dataset root with `train/images`, `train/labels`, `val/`, `test/`, and a `data.yaml` (path, train/val/test, `nc: 10`, class names).

---

## Model Selection & Training

| Model     | Architecture | Description |
|----------|--------------|-------------|
| **YOLO26** | CNN          | Efficient baseline. Single-stage CNN detector; pretrained on COCO, fine-tuned on VisDrone. |
| **RF-DETR** | Transformer  | Advanced model. Transformer-based (DETR-style) detector; base variant fine-tuned on VisDrone (COCO format). |

**Training (from repo root):**

```bash
# YOLO26: pass path to data.yaml
python scripts/train_yolo26.py --data <path/to/data.yaml> --epochs 50

# RF-DETR: convert YOLO to COCO, then train
python scripts/convert_yolo_to_coco.py --input <yolo_dataset_root> --output <coco_output_root>
python scripts/train_rfdetr.py --dataset-dir <coco_output_root> --output-dir <run_output_dir> --epochs 10
```

Weights are written to the paths or default run directories specified. Hyperparameters (epochs, batch size, lr) are set in the scripts and tunable via CLI.

---

## Evaluation & Error Analysis

- **Splits**: Train, validation, test (VisDrone standard; test used only for final metrics)
- **Metrics**: mAP50, mAP50-95, precision, recall; per-class AP where available. RF-DETR evaluation uses COCO-style AP
- **Running Tests**:
  - YOLO26: `python scripts/test_yolo26.py --data <path/to/data.yaml> --output-dir <output_dir> --save-predictions` — writes metrics, plots, and annotated test images to the output directory
  - RF-DETR: `python scripts/test_rfdetr.py --dataset-dir <coco_root> --output-dir <output_dir> --save-annotated` — writes metrics and annotated test images
- **Unified Test Run**: `python scripts/run_test_suite.py --data <path/to/data.yaml> --data-coco <coco_root> --output-dir <base_output_dir>` runs both models and writes results in a standard layout
- **VisDrone2019-DET-test-dev**: Place the **VisDrone2019-DET-test-dev** folder (with `images/` and `annotations/` inside) in the repo root or in a directory passed as `--testdev-dir`. Run `python scripts/run_test_on_testdev.py --testdev-dir <dir> --output-dir test_results_testdev` to convert test-dev to YOLO and COCO, run both models, and write to `test_results_testdev/results/yolo26/` and `test_results_testdev/results/rfdetr/`
- **Error Analysis**: `python scripts/error_analysis.py --model yolo26` or `--model rfdetr` (with appropriate `--weights` and paths) to inspect validation metrics and failure samples
- **Typical Failure Modes**: Small or distant objects, heavy occlusion, confusion between similar classes (e.g. car/van, pedestrian/people)
- **Improvement Directions**: More data or small-object augmentation; higher input resolution; class-weighted loss or oversampling; post-processing (e.g. NMS tuning)

---

## Model Performance & Test Results (VisDrone2019-DET-test-dev)

Evaluation is run on **VisDrone2019-DET-test-dev** (1,610 images, 75,102 instances). Both models are tested and results are saved in a standard layout under `test_results_testdev/results/`.

### Commands to Reproduce

```bash
# Full pipeline (convert test-dev → YOLO/COCO, then run both models)
python scripts/run_test_on_testdev.py --testdev-dir . --output-dir test_results_testdev

# Test only (dataset already converted)
python scripts/run_test_on_testdev.py --test-only --output-dir test_results_testdev

# RF-DETR only (YOLO26 results already saved)
python scripts/run_test_on_testdev.py --test-only --rfdetr-only --output-dir test_results_testdev
```

### Output Layout

The evaluation pipeline generates comprehensive analysis outputs that demonstrate the effectiveness of both models. Each of `test_results_testdev/results/yolo26/` and `test_results_testdev/results/rfdetr/` contains:

| Item | Description |
|------|--------------|
| `metrics.json` | mAP50, mAP50-95, precision, recall; COCO-style AP for RF-DETR |
| `plots/` | Precision–recall curves, confusion matrix, validation batch predictions |
| `annotated/` | Test images with predicted bounding boxes drawn |

### Test Set Results (YOLO26)

| Metric | Value |
|--------|--------|
| mAP50 | 0.334 |
| mAP50-95 | 0.191 |
| mAP75 | 0.193 |
| Precision | 0.454 |
| Recall | 0.348 |

Per-class mAP50 (VisDrone 10 classes): car (0.47), bus (0.38), van (0.25), truck (0.24), pedestrian (0.13), motor (0.13), awning-tricycle (0.11), tricycle (0.10), people (0.06), bicycle (0.05).

### Test Set Results (RF-DETR)

COCO-style AP metrics are written to `test_results_testdev/results/rfdetr/metrics.json` (AP, AP50, AP75) when the RF-DETR test is run with a valid checkpoint.

### Training Results and Visualizations

The following visualizations are produced under `test_results_testdev/results/yolo26/plots/`. All paths are relative to the repository root.

#### Confusion Matrix

![YOLO26 confusion matrix](test_results_testdev/results/yolo26/plots/confusion_matrix_normalized.png)
*Normalized confusion matrix showing classification performance across all 10 VisDrone classes*

#### Precision-Recall Curve

![YOLO26 PR curve](test_results_testdev/results/yolo26/plots/BoxPR_curve.png)
*Precision-Recall curve demonstrating detection quality at various confidence thresholds*

#### Precision Curve

![YOLO26 P curve](test_results_testdev/results/yolo26/plots/BoxP_curve.png)
*Precision curve showing model precision across confidence thresholds*

#### Sample Predictions

![val batch0 pred](test_results_testdev/results/yolo26/plots/val_batch0_pred.jpg)
*Sample validation batch predictions showing detected objects with bounding boxes and class labels*

![val batch1 pred](test_results_testdev/results/yolo26/plots/val_batch1_pred.jpg)
*Additional validation batch predictions demonstrating model performance on drone-captured scenes*

Annotated test images (model predictions drawn on test-dev images) are saved in:
- `test_results_testdev/results/yolo26/annotated/`
- `test_results_testdev/results/rfdetr/annotated/`

### Analysis Outputs Explained

The `test_results_testdev/results/` folder contains comprehensive visualizations and metrics:

- **`yolo26/metrics.json`**: mAP50, mAP50-95, precision, recall, and per-class AP
- **`yolo26/plots/confusion_matrix_normalized.png`**: Classification performance per class
- **`yolo26/plots/BoxPR_curve.png`** and **`BoxP_curve.png`**: Detection quality curves
- **`yolo26/plots/val_batch*_pred.jpg`**: Validation batch prediction visualizations
- **`yolo26/annotated/`** and **`rfdetr/annotated/`**: Test images with predicted bounding boxes drawn

These outputs demonstrate the model evaluation process, metric reporting, and visual inspection capabilities for both YOLO26 and RF-DETR on the VisDrone test-dev set.

---

## Inference & Deployment

**CLI:**

```bash
python scripts/inference.py --model yolo26 --image <path/to/image.jpg> [--out <output.jpg>]
python scripts/inference.py --model rfdetr --image <path/to/image.jpg> [--out <output.jpg>]
```

**Inference API with UI:**

- The API allows switching between **YOLO26** and **RF-DETR**; uploading **image or video**; running inference; and receiving **annotated image/video** and detections (for images)
- Run from repo root:

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- Open **http://localhost:8000** in a browser: select model, choose file, click "Run inference", then view or download the result
- **API**: `POST /infer` with form fields `model` (yolo26 | rfdetr) and `file` (image or video). Response includes `result_url` (annotated file) and `detections` (for images)

Model loading and paths are configurable; default weights are resolved from the `models/` directory (see `api/engine.py`).

---

## File Structure

```
VisDrone_YOLO-RF-Detr/
├── scripts/                          # Data, training, testing, inference
│   ├── convert_visdrone_to_yolo.py   # VisDrone → YOLO format
│   ├── convert_yolo_to_coco.py       # YOLO → COCO for RF-DETR
│   ├── check_dataset.py              # Dataset validation
│   ├── analyze_data.py               # Class distribution, counts
│   ├── train_yolo26.py               # YOLO26 training
│   ├── train_rfdetr.py               # RF-DETR training
│   ├── test_yolo26.py                # YOLO26 evaluation
│   ├── test_rfdetr.py                # RF-DETR evaluation
│   ├── run_test_suite.py             # Run both models, standard output
│   ├── run_test_on_testdev.py        # Full test-dev pipeline (convert + test)
│   ├── inference.py                  # CLI inference
│   └── error_analysis.py             # Validation analysis
├── api/                               # Inference API and UI
│   ├── main.py                       # FastAPI app
│   ├── engine.py                     # YOLO26 / RF-DETR inference
│   └── static/index.html             # Web UI (upload, run, download)
├── models/                            # Weights and artifacts
│   ├── Yolo26/weights/               # best.pt
│   └── rfdetr/weights/               # checkpoint_best_regular.pth, etc.
├── test_results_testdev/              # Test-dev run output (after running pipeline)
│   ├── yolo_data/                    # YOLO format (test/images, test/labels, data.yaml)
│   ├── coco_data/                    # COCO format (test/)
│   └── results/
│       ├── yolo26/                   # YOLO26 test results
│       │   ├── metrics.json          # mAP50, mAP50-95, precision, recall
│       │   ├── plots/                # Confusion matrix, PR curves, val batch preds
│       │   └── annotated/            # Test images with predicted boxes
│       └── rfdetr/                   # RF-DETR test results
│           ├── metrics.json          # COCO AP, AP50, AP75
│           ├── plots/                # Metrics summary plot
│           └── annotated/            # Test images with predicted boxes
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Data: place VisDrone (YOLO format) in a dataset directory, then:
python scripts/check_dataset.py --data-root <dataset_root>
python scripts/analyze_data.py --data <dataset_root>
# Ensure a data.yaml exists (path, train/val/test, nc: 10, names)

# 3. Train (optional; use provided weights under models/ if available)
python scripts/train_yolo26.py --data <path/to/data.yaml> --epochs 50
python scripts/convert_yolo_to_coco.py --input <yolo_dataset_root> --output <coco_output_root>
python scripts/train_rfdetr.py --dataset-dir <coco_output_root> --output-dir <run_output_dir> --epochs 10

# 4. Test (single model or both in standard layout)
python scripts/test_yolo26.py --data <path/to/data.yaml> --output-dir <output_dir> --save-predictions
python scripts/test_rfdetr.py --dataset-dir <coco_root> --output-dir <output_dir> --save-annotated
# Or run both:
python scripts/run_test_suite.py --data <path/to/data.yaml> --data-coco <coco_root> --output-dir test_results

# 5. Inference API + UI
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000 in a browser
```

---

## Performance Highlights

- **Dual Model Evaluation**: YOLO26 (CNN baseline) and RF-DETR (transformer) evaluated on VisDrone2019-DET-test-dev
- **Standard Result Layout**: Metrics, plots (confusion matrix, PR curves), and annotated test images per model
- **Reproducible Pipeline**: Single script converts test-dev to YOLO/COCO and runs both tests; optional test-only or RF-DETR-only runs
- **Inference API**: Web UI to switch models, upload image/video, and download annotated output

This project represents a complete pipeline for drone-view object detection, from dataset preparation and model training to evaluation and deployment with both CNN and transformer-based detectors.

---

## License & References

- **VisDrone**: [visdrone.net](http://aiskyeye.com/)
- **YOLO**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **RF-DETR**: [Roboflow RF-DETR](https://github.com/roboflow/rf-detr)

---

## Summary

This repository represents a complete, production-ready implementation of object detection on the VisDrone benchmark using two detector families: an efficient CNN baseline (YOLO26) and an advanced transformer-based model (RF-DETR). The project demonstrates a comprehensive approach from data preparation and model training to evaluation and deployment, including an inference API with a web UI.

**What makes this repository special:**

The system combines two complementary detection paradigms in one pipeline: **YOLO26** (fast, CNN-based, single-stage) and **RF-DETR** (transformer-based, set prediction). This dual-model approach is valuable for comparing efficiency versus accuracy and for deployment flexibility (e.g. lightweight YOLO26 for real-time use, RF-DETR for higher accuracy when compute allows).

**Technical Excellence:**
- **Dual Architecture**: CNN baseline and transformer-based detector with shared evaluation and deployment tooling
- **Complete Pipeline**: Data conversion (VisDrone → YOLO → COCO), training scripts, unified test pipeline (test-dev), and inference API
- **Standard Outputs**: Metrics (JSON), plots (confusion matrix, PR curves), and annotated images in a consistent layout per model
- **Documentation**: Step-by-step guidance from dataset setup to training, testing, and deployment

**Real-World Impact:**

With evaluation on the official VisDrone2019-DET-test-dev set (1,610 images), the pipeline provides reproducible metrics and visualizations for both models. The inference API enables rapid prototyping and demos by switching between YOLO26 and RF-DETR and processing images or videos through a simple web interface.

**Complete Implementation:**

Unlike many projects that focus only on a single model or training step, this repository provides a full pipeline: dataset validation and conversion, training for both YOLO26 and RF-DETR, unified test-dev evaluation with saved metrics and plots, CLI inference, and a FastAPI-based web UI for inference. All paths and commands are documented so that the entire process from raw VisDrone data to deployed inference can be reproduced.

This project serves as a practical example of how to build, train, evaluate, and deploy object detection models on drone-captured imagery, making it useful for researchers, practitioners, and students interested in VisDrone or in comparing CNN and transformer-based detectors.
