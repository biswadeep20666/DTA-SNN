"""Evaluation script for DTA-SNN on ETram test set."""
import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent))

from models.dta_snn_seq2seq import build_dta_snn_seq2seq
from datasets.etram_dataset import build_etram_dataloader


def focal_loss(pred, target, alpha=0.75, gamma=2.0):
    """Focal loss for binary event prediction."""
    pred = pred.clamp(1e-6, 1 - 1e-6)
    bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    p_t = target * pred + (1 - target) * (1 - pred)
    focal_weight = (1 - p_t) ** gamma
    
    if alpha >= 0:
        alpha_t = target * alpha + (1 - target) * (1 - alpha)
        focal_weight = alpha_t * focal_weight
    
    return (focal_weight * bce).mean()


def compute_detailed_metrics(pred, target):
    """Compute comprehensive evaluation metrics."""
    with torch.no_grad():
        # Regression metrics
        mse = nn.functional.mse_loss(pred, target).item()
        mae = nn.functional.l1_loss(pred, target).item()
        
        # Binary predictions
        pred_binary = (pred > 0.5).float()
        target_binary = (target > 0.5).float()
        
        # Per-pixel classification metrics
        tp = (pred_binary * target_binary).sum().item()
        fp = (pred_binary * (1 - target_binary)).sum().item()
        fn = ((1 - pred_binary) * target_binary).sum().item()
        tn = ((1 - pred_binary) * (1 - target_binary)).sum().item()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        
        # Per-frame metrics (averaged over batch and time)
        B, T = pred.shape[:2]
        frame_mse = []
        frame_f1 = []
        
        for b in range(B):
            for t in range(T):
                p_frame = pred[b, t]
                y_frame = target[b, t]
                frame_mse.append(nn.functional.mse_loss(p_frame, y_frame).item())
                
                p_bin = (p_frame > 0.5).float()
                y_bin = (y_frame > 0.5).float()
                tp_f = (p_bin * y_bin).sum().item()
                fp_f = (p_bin * (1 - y_bin)).sum().item()
                fn_f = ((1 - p_bin) * y_bin).sum().item()
                prec_f = tp_f / (tp_f + fp_f + 1e-8)
                rec_f = tp_f / (tp_f + fn_f + 1e-8)
                f1_f = 2 * prec_f * rec_f / (prec_f + rec_f + 1e-8)
                frame_f1.append(f1_f)
        
        return {
            "mse": mse,
            "mae": mae,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "frame_mse_mean": np.mean(frame_mse),
            "frame_mse_std": np.std(frame_mse),
            "frame_f1_mean": np.mean(frame_f1),
            "frame_f1_std": np.std(frame_f1),
        }


@torch.no_grad()
def evaluate(model, dataloader, device, use_focal=True, 
             focal_alpha=0.75, focal_gamma=2.0, use_amp=True):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0.0
    all_metrics = []
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(inputs)
            
            if use_focal:
                loss = focal_loss(outputs, targets, alpha=focal_alpha, gamma=focal_gamma)
            else:
                loss = nn.functional.mse_loss(outputs, targets)
        
        metrics = compute_detailed_metrics(outputs, targets)
        metrics["loss"] = loss.item()
        
        total_loss += loss.item()
        all_metrics.append(metrics)
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Aggregate metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        avg_metrics[key + "_std"] = np.std([m[key] for m in all_metrics])
    
    return avg_metrics, all_metrics


def visualize_predictions(model, dataloader, device, save_dir, num_samples=5):
    """Visualize sample predictions."""
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            
            # Visualize first sample in batch
            inp = inputs[0].cpu().numpy()      # [T, 2, H, W]
            tgt = targets[0].cpu().numpy()     # [T, 2, H, W]
            out = outputs[0].cpu().numpy()     # [T, 2, H, W]
            
            # Create grid: input | target | prediction
            T = tgt.shape[0]
            fig, axes = plt.subplots(3, T, figsize=(T * 2, 6))
            
            for t in range(T):
                # Combine ON (red) and OFF (blue) channels
                inp_vis = np.stack([inp[-1, 0], np.zeros_like(inp[-1, 0]), inp[-1, 1]], axis=-1)
                tgt_vis = np.stack([tgt[t, 0], np.zeros_like(tgt[t, 0]), tgt[t, 1]], axis=-1)
                out_vis = np.stack([out[t, 0], np.zeros_like(out[t, 0]), out[t, 1]], axis=-1)
                
                axes[0, t].imshow(inp_vis if t == 0 else tgt_vis)
                axes[0, t].set_title(f"Input t={t}" if t == 0 else f"GT t={t}")
                axes[0, t].axis("off")
                
                axes[1, t].imshow(tgt_vis)
                axes[1, t].set_title(f"Target t={t}")
                axes[1, t].axis("off")
                
                axes[2, t].imshow(out_vis)
                axes[2, t].set_title(f"Pred t={t}")
                axes[2, t].axis("off")
            
            plt.tight_layout()
            plt.savefig(save_dir / f"sample_{batch_idx:03d}.png", dpi=150, bbox_inches="tight")
            plt.close()
    
    print(f"Saved {num_samples} visualization samples to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DTA-SNN on ETram test set")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, 
                       default="/home/biswadeep/OpenSTL/data/etram_npz_binary_30hz_crop512_runs",
                       help="Path to dataset root")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--device", type=str, default="cuda:1",
                       help="Device to use")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Directory to save results (default: same as checkpoint)")
    parser.add_argument("--visualize", action="store_true", default=True,
                       help="Generate visualization samples")
    parser.add_argument("--num_vis_samples", type=int, default=5,
                       help="Number of samples to visualize")
    
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint.get("args", {})
    
    # Setup save directory
    if args.save_dir is None:
        args.save_dir = checkpoint_path.parent / "evaluation"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("Evaluating DTA-SNN on ETram Test Set")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Results will be saved to: {save_dir}")
    print("=" * 80)
    
    # Build test dataloader
    print("\nBuilding test dataloader...")
    test_loader = build_etram_dataloader(
        args.data_root, "test",
        pre_seq=train_args.get("pre_seq", 10),
        aft_seq=train_args.get("aft_seq", 10),
        batch_size=args.batch_size,
        stride=1,
        num_workers=args.num_workers,
        use_obj_mask=train_args.get("use_obj_mask", False),
    )
    
    # Build model
    print("\nBuilding model...")
    model = build_dta_snn_seq2seq(
        pre_seq=train_args.get("pre_seq", 10),
        aft_seq=train_args.get("aft_seq", 10),
        in_channels=2,
        out_channels=2,
        encoder_time=train_args.get("pre_seq", 10),
        encoder_layers=tuple(train_args.get("encoder_layers", [3, 3, 2])),
        bottleneck_channels=train_args.get("bottleneck_channels", 256),
        DTA_ON=train_args.get("DTA_ON", True),
        activation="sigmoid",
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Evaluate
    print("\nEvaluating on test set...")
    avg_metrics, all_metrics = evaluate(
        model, test_loader, device,
        use_focal=train_args.get("use_focal", True),
        focal_alpha=train_args.get("focal_alpha", 0.75),
        focal_gamma=train_args.get("focal_gamma", 2.0),
        use_amp=train_args.get("use_amp", True),
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("Test Set Results")
    print("=" * 80)
    for key, value in avg_metrics.items():
        if not key.endswith("_std"):
            std_key = key + "_std"
            if std_key in avg_metrics:
                print(f"  {key:20s}: {value:.6f} ± {avg_metrics[std_key]:.6f}")
            else:
                print(f"  {key:20s}: {value:.6f}")
    print("=" * 80)
    
    # Save results
    results = {
        "checkpoint": str(checkpoint_path),
        "test_metrics": avg_metrics,
        "train_args": train_args,
    }
    
    with open(save_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    np.savez(save_dir / "detailed_metrics.npz", metrics=all_metrics)
    
    print(f"\nResults saved to {save_dir}")
    
    # Visualize predictions
    if args.visualize:
        print(f"\nGenerating {args.num_vis_samples} visualization samples...")
        visualize_predictions(
            model, test_loader, device,
            save_dir / "visualizations",
            num_samples=args.num_vis_samples
        )
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
