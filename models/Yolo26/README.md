# YOLO26

- **Weights:** `weights/best.pt`
- **Train:** `python scripts/train_yolo26.py --data data/visdrone/data.yaml --epochs 50`
- **Test:** `python scripts/test_yolo26.py --weights models/Yolo26/weights/best.pt`
- **Inference:** `python scripts/inference.py --model yolo26 --image path/to/image.jpg`
