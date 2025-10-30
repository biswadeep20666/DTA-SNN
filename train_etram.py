"""Training script for DTA-SNN on ETram dataset."""
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent))

from models.dta_snn_seq2seq import build_dta_snn_seq2seq
from datasets.etram_dataset import build_etram_dataloader


def focal_loss(pred, target, alpha=0.75, gamma=2.0):
    """
    Focal loss for binary event prediction.
    
    Args:
        pred: Predicted probabilities [B, T, C, H, W] in [0, 1]
        target: Ground truth [B, T, C, H, W] in [0, 1]
        alpha: Class balance weight (0.75 means more weight to positive class)
        gamma: Focusing parameter (higher = more focus on hard examples)
    
    Returns:
        Scalar loss value
    """
    pred = pred.clamp(1e-6, 1 - 1e-6)
    bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    p_t = target * pred + (1 - target) * (1 - pred)
    focal_weight = (1 - p_t) ** gamma
    
    if alpha >= 0:
        alpha_t = target * alpha + (1 - target) * (1 - alpha)
        focal_weight = alpha_t * focal_weight
    
    return (focal_weight * bce).mean()


def compute_metrics(pred, target):
    """
    Compute evaluation metrics for binary event prediction.
    
    Args:
        pred: Predictions [B, T, C, H, W]
        target: Ground truth [B, T, C, H, W]
    
    Returns:
        Dictionary of metrics
    """
    with torch.no_grad():
        # MSE
        mse = nn.functional.mse_loss(pred, target).item()
        
        # MAE
        mae = nn.functional.l1_loss(pred, target).item()
        
        # Binary predictions (threshold at 0.5)
        pred_binary = (pred > 0.5).float()
        target_binary = (target > 0.5).float()
        
        # Accuracy
        acc = (pred_binary == target_binary).float().mean().item()
        
        # F1 score (per-pixel)
        tp = (pred_binary * target_binary).sum().item()
        fp = (pred_binary * (1 - target_binary)).sum().item()
        fn = ((1 - pred_binary) * target_binary).sum().item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            "mse": mse,
            "mae": mae,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def train_epoch(model, dataloader, optimizer, device, use_focal=True, 
                focal_alpha=0.75, focal_gamma=2.0, use_amp=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_metrics = {}
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    pbar = tqdm(dataloader, desc="Training")
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)    # [B, pre_seq, 2, H, W]
        targets = targets.to(device, non_blocking=True)  # [B, aft_seq, 2, H, W]
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(inputs)  # [B, aft_seq, 2, H, W]
            
            if use_focal:
                loss = focal_loss(outputs, targets, alpha=focal_alpha, gamma=focal_gamma)
            else:
                loss = nn.functional.mse_loss(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Compute metrics
        metrics = compute_metrics(outputs.detach(), targets)
        
        total_loss += loss.item()
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "f1": f"{metrics['f1']:.4f}",
        })
    
    n_batches = len(dataloader)
    avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}
    avg_metrics["loss"] = total_loss / n_batches
    
    return avg_metrics


@torch.no_grad()
def validate(model, dataloader, device, use_focal=True, 
             focal_alpha=0.75, focal_gamma=2.0, use_amp=True):
    """Validate on validation set."""
    model.eval()
    total_loss = 0.0
    total_metrics = {}
    
    pbar = tqdm(dataloader, desc="Validation")
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(inputs)
            
            if use_focal:
                loss = focal_loss(outputs, targets, alpha=focal_alpha, gamma=focal_gamma)
            else:
                loss = nn.functional.mse_loss(outputs, targets)
        
        metrics = compute_metrics(outputs, targets)
        
        total_loss += loss.item()
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    n_batches = len(dataloader)
    avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}
    avg_metrics["loss"] = total_loss / n_batches
    
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train DTA-SNN on ETram dataset")
    
    # Data args
    parser.add_argument("--data_root", type=str, 
                       default="/home/biswadeep/OpenSTL/data/etram_npz_binary_30hz_crop512_runs",
                       help="Path to dataset root")
    parser.add_argument("--use_obj_mask", action="store_true", default=False,
                       help="Apply object masks to history frames")
    
    # Model args
    parser.add_argument("--pre_seq", type=int, default=10,
                       help="Number of input frames")
    parser.add_argument("--aft_seq", type=int, default=10,
                       help="Number of prediction frames")
    parser.add_argument("--DTA_ON", action="store_true", default=True,
                       help="Enable DTA module")
    parser.add_argument("--encoder_layers", type=int, nargs=3, default=[3, 3, 2],
                       help="Number of blocks in each encoder layer")
    parser.add_argument("--bottleneck_channels", type=int, default=256,
                       help="Bottleneck channels")
    
    # Training args
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay")
    parser.add_argument("--use_focal", action="store_true", default=True,
                       help="Use focal loss instead of MSE")
    parser.add_argument("--focal_alpha", type=float, default=0.75,
                       help="Focal loss alpha (class balance)")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                       help="Focal loss gamma (focusing parameter)")
    parser.add_argument("--use_amp", action="store_true", default=True,
                       help="Use automatic mixed precision")
    
    # Dataloader args
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--train_stride", type=int, default=1,
                       help="Stride for training window extraction")
    parser.add_argument("--val_stride", type=int, default=5,
                       help="Stride for validation window extraction")
    
    # System args
    parser.add_argument("--device", type=str, default="cuda:1",
                       help="Device to use")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/etram_dta_snn",
                       help="Directory to save checkpoints")
    parser.add_argument("--log_interval", type=int, default=1,
                       help="Epochs between logging")
    parser.add_argument("--save_interval", type=int, default=10,
                       help="Epochs between checkpoint saves")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save args
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print("=" * 80)
    print("Training DTA-SNN on ETram Dataset")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")
    print(f"Input/Output: {args.pre_seq} → {args.aft_seq} frames")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Loss: {'Focal' if args.use_focal else 'MSE'}")
    print(f"DTA: {'ON' if args.DTA_ON else 'OFF'}")
    print(f"Object masking: {'ON' if args.use_obj_mask else 'OFF'}")
    print(f"Mixed precision: {'ON' if args.use_amp else 'OFF'}")
    print("=" * 80)
    
    # Dataloaders
    print("\nBuilding dataloaders...")
    train_loader = build_etram_dataloader(
        args.data_root, "train", args.pre_seq, args.aft_seq, 
        args.batch_size, stride=args.train_stride, 
        num_workers=args.num_workers, use_obj_mask=args.use_obj_mask
    )
    val_loader = build_etram_dataloader(
        args.data_root, "val", args.pre_seq, args.aft_seq,
        args.batch_size, stride=args.val_stride, 
        num_workers=args.num_workers, use_obj_mask=args.use_obj_mask
    )
    
    # Model
    print("\nBuilding model...")
    model = build_dta_snn_seq2seq(
        pre_seq=args.pre_seq,
        aft_seq=args.aft_seq,
        in_channels=2,
        out_channels=2,
        encoder_time=args.pre_seq,
        encoder_layers=tuple(args.encoder_layers),
        bottleneck_channels=args.bottleneck_channels,
        DTA_ON=args.DTA_ON,
        activation="sigmoid",
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Optimizer & Scheduler
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=args.lr / 100
    )
    
    # Training loop
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    train_history = []
    val_history = []
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            use_focal=args.use_focal,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            use_amp=args.use_amp,
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, device,
            use_focal=args.use_focal,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            use_amp=args.use_amp,
        )
        
        scheduler.step()
        
        # Log
        if epoch % args.log_interval == 0:
            print(f"\n[Epoch {epoch}] Results:")
            print(f"  Train - Loss: {train_metrics['loss']:.6f} | "
                  f"F1: {train_metrics['f1']:.4f} | "
                  f"Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.6f} | "
                  f"F1: {val_metrics['f1']:.4f} | "
                  f"Acc: {val_metrics['accuracy']:.4f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save history
        train_history.append(train_metrics)
        val_history.append(val_metrics)
        
        # Save best model (by loss)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "args": vars(args),
            }, save_dir / "best_model_loss.pth")
            print(f"  ✓ Saved best model (loss={val_metrics['loss']:.6f})")
        
        # Save best model (by F1)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "args": vars(args),
            }, save_dir / "best_model_f1.pth")
            print(f"  ✓ Saved best model (F1={val_metrics['f1']:.4f})")
        
        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }, save_dir / f"checkpoint_epoch{epoch:03d}.pth")
            print(f"  ✓ Saved checkpoint at epoch {epoch}")
        
        # Save training history
        np.savez(
            save_dir / "training_history.npz",
            train=train_history,
            val=val_history,
        )
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Checkpoints saved to: {save_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
