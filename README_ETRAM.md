# DTA-SNN Training on ETram Dataset

Complete pipeline for training and evaluating DTA-SNN on the ETram event-based dataset.

## Setup

```bash
cd /home/biswadeep/DTA-SNN
```

## Dataset Structure

Expected dataset structure:
```
/home/biswadeep/OpenSTL/data/etram_npz_binary_30hz_crop512_runs/
├── train/
│   ├── train_day_0001_1.npz
│   ├── train_day_0001_2.npz
│   └── ...
├── val/
│   ├── val_day_001_1.npz
│   └── ...
└── test/
    ├── test_day_001_1.npz
    └── ...
```

Each `.npz` file contains:
- `frames`: [T, 2, H, W] uint8 event frames (ON/OFF channels)
- `obj_mask`: [T, H, W] optional object masks
- `present_mask`: [T] boolean presence per frame
- Metadata: crop params, timestamps, etc.

## Training

### Quick Start (Default Settings)

Train on GPU 1 with default hyperparameters:

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Train with background events
bash scripts/train_etram.sh
```

### Training with Object Masking

To apply object masks to history frames (removes background noise, no label leakage):

```bash
bash scripts/train_etram_objmask.sh
```

### Manual Training with Custom Settings

```bash
CUDA_VISIBLE_DEVICES=1 python train_etram.py \
    --data_root /home/biswadeep/OpenSTL/data/etram_npz_binary_30hz_crop512_runs \
    --save_dir ./checkpoints/my_experiment \
    --pre_seq 10 \
    --aft_seq 10 \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --use_focal \
    --focal_alpha 0.75 \
    --focal_gamma 2.0 \
    --DTA_ON \
    --use_amp \
    --num_workers 4 \
    --device cuda:1
```

### Key Arguments

**Data:**
- `--data_root`: Path to dataset root
- `--use_obj_mask`: Apply object masks to history frames (no label leakage)

**Model:**
- `--pre_seq`: Number of input frames (default: 10)
- `--aft_seq`: Number of prediction frames (default: 10)
- `--DTA_ON`: Enable Dual Temporal Attention module
- `--encoder_layers`: Number of blocks per encoder layer (default: 3 3 2)
- `--bottleneck_channels`: Bottleneck dimension (default: 256)

**Training:**
- `--batch_size`: Batch size (default: 4, increase to 8 or 16 if memory allows)
- `--epochs`: Number of epochs (default: 50)
- `--lr`: Learning rate (default: 1e-3)
- `--use_focal`: Use focal loss instead of MSE (recommended for sparse events)
- `--focal_alpha`: Focal loss class balance (0.75 = more weight to events)
- `--focal_gamma`: Focal loss focusing parameter (2.0 = focus on hard examples)
- `--use_amp`: Mixed precision training (faster, less memory)

**System:**
- `--device`: GPU device (default: cuda:1)
- `--num_workers`: DataLoader workers (default: 4)
- `--save_dir`: Checkpoint directory

## Evaluation

Evaluate a trained model on the test set:

```bash
# Using the evaluation script
bash scripts/eval_etram.sh checkpoints/etram_dta_snn_XXXXX/best_model_f1.pth

# Or manually
CUDA_VISIBLE_DEVICES=1 python evaluate_etram.py \
    --checkpoint checkpoints/etram_dta_snn_XXXXX/best_model_f1.pth \
    --data_root /home/biswadeep/OpenSTL/data/etram_npz_binary_30hz_crop512_runs \
    --batch_size 8 \
    --device cuda:1 \
    --visualize \
    --num_vis_samples 10
```

This will:
- Compute metrics: MSE, MAE, Accuracy, Precision, Recall, F1, IoU
- Save results to `checkpoints/.../evaluation/test_results.json`
- Generate visualization samples if `--visualize` is set

## Monitoring Training

### View Training Progress

```bash
# If running in background with nohup
tail -f nohup.out

# Or check saved training history
python -c "import numpy as np; h = np.load('checkpoints/YOUR_EXP/training_history.npz', allow_pickle=True); print(h['val'][-1])"
```

### TensorBoard (Optional)

To add TensorBoard logging, install it and modify `train_etram.py`:

```bash
pip install tensorboard

# Add to train_etram.py after imports:
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=save_dir / "logs")

# Add after computing metrics:
writer.add_scalar("train/loss", train_metrics["loss"], epoch)
writer.add_scalar("val/loss", val_metrics["loss"], epoch)
# ... etc

# Run TensorBoard
tensorboard --logdir checkpoints --port 6006
```

## Expected Results

**Baseline (no object masking):**
- F1: ~0.4-0.6 (depends on background noise)
- IoU: ~0.3-0.5
- MSE: ~0.01-0.03

**With object masking:**
- F1: ~0.6-0.8 (cleaner signal)
- IoU: ~0.5-0.7
- MSE: ~0.005-0.02

Training time: ~2-4 hours for 100 epochs on RTX 3090

## Directory Structure After Training

```
DTA-SNN/
├── datasets/
│   ├── __init__.py
│   └── etram_dataset.py
├── models/
│   ├── dta_snn_seq2seq.py
│   └── ...
├── scripts/
│   ├── train_etram.sh
│   ├── train_etram_objmask.sh
│   └── eval_etram.sh
├── checkpoints/
│   └── etram_dta_snn_YYYYMMDD_HHMMSS/
│       ├── args.json
│       ├── best_model_loss.pth
│       ├── best_model_f1.pth
│       ├── checkpoint_epoch010.pth
│       ├── training_history.npz
│       └── evaluation/
│           ├── test_results.json
│           ├── detailed_metrics.npz
│           └── visualizations/
│               ├── sample_000.png
│               └── ...
├── train_etram.py
├── evaluate_etram.py
└── README_ETRAM.md
```

## Troubleshooting

### Out of Memory

Reduce batch size or enable gradient checkpointing:

```bash
python train_etram.py --batch_size 2 --use_amp
```

### Slow Training

Increase num_workers or reduce validation frequency:

```bash
python train_etram.py --num_workers 8 --val_stride 10
```

### No Valid Windows in Dataset

Check if dataset generation completed successfully:

```bash
python -c "
import numpy as np
from pathlib import Path
root = Path('/home/biswadeep/OpenSTL/data/etram_npz_binary_30hz_crop512_runs')
for split in ['train', 'val', 'test']:
    files = list((root / split).glob('*.npz'))
    total_frames = sum(np.load(f)['frames'].shape[0] for f in files)
    print(f'{split}: {len(files)} files, {total_frames} frames')
"
```

## Citation

If you use this code, please cite:

```bibtex
@article{dtasnn2024,
  title={Direct Training of Spiking Neural Networks with Surrogate Gradient},
  author={...},
  journal={...},
  year={2024}
}
```

## Contact

For issues or questions, please contact the repository maintainer.
