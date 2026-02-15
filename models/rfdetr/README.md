# RF-DETR

- **Weights:** `weights/` (e.g. `checkpoint_best_ema.pth`)
- **Train:** `python scripts/train_rfdetr.py --dataset-dir data/visdrone_coco --output-dir runs/rfdetr --epochs 10`
- **Test:** `python scripts/test_rfdetr.py --checkpoint models/rfdetr/weights/checkpoint_best_ema.pth --dataset-dir data/visdrone_coco`
- **Inference:** `python scripts/inference.py --model rfdetr --image path/to/image.jpg`
