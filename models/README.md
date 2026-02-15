# Models

Weights and artifacts for **YOLO26** and **RF-DETR** (VisDrone). All training, evaluation, and inference are run from repo root via `scripts/`.

## Structure

| Folder    | Contents |
|-----------|----------|
| **Yolo26/** | `weights/best.pt`, training artifacts (args.yaml, results.csv, plots) |
| **rfdetr/** | `weights/*.pth`, `evaluation_plots/`, `log.txt` |

## Commands (run from repo root)

```bash
# Inference (default weights: models/...)
python scripts/inference.py --model yolo26 --image path/to/image.jpg
python scripts/inference.py --model rfdetr --image path/to/image.jpg

# Train
python scripts/train_yolo26.py --data data/visdrone/data.yaml --epochs 50
python scripts/train_rfdetr.py --dataset-dir data/visdrone_coco --output-dir runs/rfdetr --epochs 10

# Test (metrics + plots)
python scripts/test_yolo26.py --weights models/Yolo26/weights/best.pt
python scripts/test_rfdetr.py --checkpoint models/rfdetr/weights/checkpoint_best_ema.pth --dataset-dir data/visdrone_coco
```
