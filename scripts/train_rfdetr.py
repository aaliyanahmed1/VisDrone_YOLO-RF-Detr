from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RFDETR_BASE = "runs/rfdetr"


def print_header(text: str) -> None:
    """Print a formatted header for clean logs"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text: str) -> None:
    """Print a formatted section for clean logs"""
    print(f"\n{'─' * 70}")
    print(f"  {text}")
    print('─' * 70)


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest checkpoint in the output directory"""
    output_path = Path(output_dir)
    
    # Look for common checkpoint patterns
    checkpoint_patterns = [
        "checkpoint_best_total.pth",
        "checkpoint_best_ema.pth",
        "checkpoint_best.pth",
        "checkpoint_last.pth",
    ]
    
    for pattern in checkpoint_patterns:
        checkpoint_path = output_path / pattern
        if checkpoint_path.exists():
            return str(checkpoint_path)
    
    # If no standard checkpoint found, look for any .pth file
    pth_files = list(output_path.glob("checkpoint_*.pth"))
    if pth_files:
        latest = max(pth_files, key=lambda p: p.stat().st_mtime)
        return str(latest)
    
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train RF-DETR on COCO-format dataset (VisDrone)"
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/rfdetr_format_dataset",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="visdrone",
    )

    # 20 epochs, stable for 14GB @ 728
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=728)

    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--resume", type=str, default="")

    args = parser.parse_args()

    # Clear screen for clean start (optional)
    print("\033[2J\033[H", end="")
    
    print_header("RF-DETR Training Pipeline")

    # Validate dataset directory
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = (REPO_ROOT / dataset_dir).resolve()

    if not dataset_dir.exists():
        print(f"\n❌ ERROR: Dataset not found at {dataset_dir}")
        return 1

    print(f"✓ Dataset directory: {dataset_dir}")

    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{RFDETR_BASE}/{args.run_name}_{stamp}"

    if not Path(output_dir).is_absolute():
        output_dir = str((REPO_ROOT / output_dir).resolve())

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")

    # Import RF-DETR
    try:
        from rfdetr import RFDETRBase
    except ImportError:
        print("\n❌ ERROR: rfdetr not installed")
        print("   Install with: pip install rfdetr")
        return 1

    print_section("Training Configuration")
    
    # Display training parameters
    config_items = [
        ("Run Name", args.run_name),
        ("Epochs", args.epochs),
        ("Batch Size", args.batch_size),
        ("Gradient Accumulation Steps", args.grad_accum_steps),
        ("Effective Batch Size", args.batch_size * args.grad_accum_steps),
        ("Learning Rate", f"{args.lr:.0e}"),
        ("Input Resolution", f"{args.resolution}x{args.resolution}"),
        ("Device", args.device if args.device else "auto-detect"),
        ("TensorBoard", "Enabled" if args.tensorboard else "Disabled"),
        ("Early Stopping", f"Enabled (patience={args.early_stopping_patience})" if args.early_stopping else "Disabled"),
        ("Checkpoint Interval", f"Every {args.checkpoint_interval} epoch(s)"),
    ]
    
    max_label_len = max(len(label) for label, _ in config_items)
    for label, value in config_items:
        print(f"  {label:<{max_label_len}} : {value}")

    # Auto-resume from latest checkpoint if no checkpoint specified
    resume_checkpoint = args.resume
    if not resume_checkpoint:
        latest_checkpoint = find_latest_checkpoint(output_dir)
        if latest_checkpoint:
            resume_checkpoint = latest_checkpoint
            print(f"\n  ℹ Auto-resume enabled: Found checkpoint at {latest_checkpoint}")
    
    if resume_checkpoint:
        print(f"\n  ⚠ Resuming from checkpoint: {resume_checkpoint}")

    print_section("Initializing Model")
    print("  Loading RF-DETR base model...")
    
    model = RFDETRBase()
    print("  ✓ Model loaded successfully")

    # Prepare training arguments
    train_kwargs = {
        "dataset_dir": str(dataset_dir),
        "output_dir": output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "resolution": args.resolution,
        "checkpoint_interval": args.checkpoint_interval,
        "tensorboard": args.tensorboard,
        "early_stopping": args.early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
        # Class-imbalance mitigation (focal loss)
        "focal_loss_alpha": 0.75,
        "focal_loss_gamma": 2.0,
    }

    if args.device:
        train_kwargs["device"] = args.device
    if resume_checkpoint:
        train_kwargs["resume"] = resume_checkpoint

    print_section("Starting Training")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if resume_checkpoint:
        print(f"  Resuming from: {resume_checkpoint}")
    print()

    try:
        model.train(**train_kwargs)
        print("\n✓ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        print("  ℹ To resume, run the same command without --resume flag")
        print(f"  Command: python scripts/train_rfdetr.py --output-dir {output_dir}")
        return 1
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        print(f"\n📋 Full error traceback:")
        print("─" * 70)
        traceback.print_exc()
        print("─" * 70)
        
        # Check for any checkpoint files
        output_path = Path(output_dir)
        pth_files = list(output_path.glob("**/*.pth"))
        if pth_files:
            latest_ckpt = max(pth_files, key=lambda p: p.stat().st_mtime)
            print(f"\n💡 Found checkpoint at: {latest_ckpt}")
            print(f"   To resume, run:")
            print(f"   python scripts/train_rfdetr.py --output-dir {output_dir} --resume {latest_ckpt}")
        else:
            print(f"\n⚠ No checkpoint files found in {output_dir}")
            print("  Training may not have saved any checkpoints yet.")
        return 1

    print_header("Training Complete")
    print(f"  ✓ Best checkpoint saved to:")
    print(f"    {output_dir}/checkpoint_best_total.pth")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())