"""Train YOLO26 on VisDrone. Load pretrained weights (e.g. weights/yolo26s.pt), train, save best.pt.

"""

import argparse
from pathlib import Path

# Repo root (parent of scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "yolo26s.pt"


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO26 on VisDrone (e.g. data=VisDrone.yaml, epochs=100, imgsz=640)"
    )
    parser.add_argument(
        "--data",
        default="data/visdrone/data.yaml",
        help="Path to data config (data/visdrone/data.yaml)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="Path to pretrained .pt model (default: weights/yolo26s.pt)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/yolo26")
    parser.add_argument("--name", default="visdrone")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (REPO_ROOT / data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_path}")

    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = (REPO_ROOT / weights_path).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}. Place yolo26s.pt in repo weights/ or set --weights.")

    from ultralytics import YOLO

    model = YOLO(str(weights_path))

    # Standard args only (this Ultralytics version does not accept fl_gamma/cls in train()).
    # For class imbalance (43:1 car vs awning-tricycle), use oversampling or a custom cfg with
    # fl_gamma and cls if your Ultralytics supports them; see docs/comments above.
    train_kwargs = {
        "data": str(data_path.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
    }
    results = model.train(**train_kwargs)
    print("Done. Best:", results.save_dir / "weights" / "best.pt")
    return 0


if __name__ == "__main__":
    exit(main())
