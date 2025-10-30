# DTA-SNN ETram Training Pipeline - Quick Start

## Files Created

### Core Training Files
1. **`datasets/etram_dataset.py`** - Dataset loader for ETram NPZ files
2. **`train_etram.py`** - Main training script
3. **`evaluate_etram.py`** - Evaluation script for test set
4. **`test_setup.py`** - Setup verification script

### Training Scripts
5. **`scripts/train_etram.sh`** - Standard training (with background)
6. **`scripts/train_etram_objmask.sh`** - Training with object masks
7. **`scripts/eval_etram.sh`** - Evaluation script

### Documentation
8. **`README_ETRAM.md`** - Complete documentation

---

## Quick Start (3 Steps)

### Step 1: Verify Setup

```bash
cd /home/biswadeep/DTA-SNN
python test_setup.py
```

This will test:
- ✓ Dataset loading from `/home/biswadeep/OpenSTL/data/etram_npz_binary_30hz_crop512_runs`
- ✓ Model instantiation
- ✓ One training step

### Step 2: Train Model on GPU 1

```bash
# Make scripts executable (only needed once)
chmod +x scripts/*.sh

# Train with default settings
bash scripts/train_etram.sh
```

**Or run in background:**

```bash
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/train_etram.sh > logs/train_etram.log 2>&1 &

# Monitor progress
tail -f logs/train_etram.log
```

### Step 3: Evaluate Trained Model

```bash
# Replace CHECKPOINT_PATH with your trained model
bash scripts/eval_etram.sh checkpoints/etram_dta_snn_XXXXX/best_model_f1.pth
```

---

## Training Configuration

**Default Settings (scripts/train_etram.sh):**
- Input: 10 frames (history)
- Output: 10 frames (prediction)
- Batch size: 8
- Epochs: 100
- Learning rate: 1e-3
- Loss: Focal loss (alpha=0.75, gamma=2.0)
- DTA: Enabled
- Mixed precision: Enabled
- GPU: cuda:1

**Expected Training Time:** ~2-4 hours on RTX 3090

---

## Customization

### Modify Hyperparameters

Edit `scripts/train_etram.sh` or run manually:

```bash
CUDA_VISIBLE_DEVICES=1 python train_etram.py \
    --data_root /home/biswadeep/OpenSTL/data/etram_npz_binary_30hz_crop512_runs \
    --save_dir ./checkpoints/my_custom_exp \
    --pre_seq 10 \
    --aft_seq 10 \
    --batch_size 16 \
    --epochs 200 \
    --lr 5e-4 \
    --use_focal \
    --DTA_ON \
    --device cuda:1
```

### Enable Object Masking

To train with object masks (removes background, no label leakage):

```bash
bash scripts/train_etram_objmask.sh
```

---

## Output Files

After training, you'll find:

```
checkpoints/etram_dta_snn_YYYYMMDD_HHMMSS/
├── args.json                    # Training arguments
├── best_model_loss.pth          # Best model by validation loss
├── best_model_f1.pth            # Best model by validation F1
├── checkpoint_epoch010.pth      # Periodic checkpoints
├── checkpoint_epoch020.pth
├── training_history.npz         # Loss/metrics history
└── evaluation/                  # Created after evaluation
    ├── test_results.json
    ├── detailed_metrics.npz
    └── visualizations/
        ├── sample_000.png
        └── ...
```

---

## Expected Results

**Metrics on Test Set:**

| Metric | With Background | With Object Masking |
|--------|----------------|---------------------|
| F1 Score | 0.4 - 0.6 | 0.6 - 0.8 |
| IoU | 0.3 - 0.5 | 0.5 - 0.7 |
| MSE | 0.01 - 0.03 | 0.005 - 0.02 |

---

## Monitoring Training

### View Real-Time Progress

```bash
# If running with nohup
tail -f logs/train_etram.log

# View specific metrics
grep "Val   - Loss" logs/train_etram.log
```

### Check Saved Metrics

```python
import numpy as np

# Load training history
history = np.load("checkpoints/YOUR_EXP/training_history.npz", allow_pickle=True)

# Print last validation metrics
print("Last validation:", history["val"][-1])

# Plot loss curve
import matplotlib.pyplot as plt
train_loss = [h["loss"] for h in history["train"]]
val_loss = [h["loss"] for h in history["val"]]
plt.plot(train_loss, label="Train")
plt.plot(val_loss, label="Val")
plt.legend()
plt.savefig("loss_curve.png")
```

---

## Troubleshooting

### Test Setup Fails

```bash
# Check dataset exists
ls /home/biswadeep/OpenSTL/data/etram_npz_binary_30hz_crop512_runs/train/*.npz | head -5

# Verify dataset format
python -c "
import numpy as np
data = np.load('/home/biswadeep/OpenSTL/data/etram_npz_binary_30hz_crop512_runs/train/train_day_0001_1.npz')
print('Keys:', list(data.keys()))
print('Frames shape:', data['frames'].shape)
"
```

### Out of Memory

Reduce batch size:

```bash
# In scripts/train_etram.sh, change:
BATCH_SIZE=4  # or 2
```

### Training Too Slow

Increase batch size or reduce validation frequency:

```bash
python train_etram.py --batch_size 16 --val_stride 10
```

---

## Next Steps

1. **Run test_setup.py** to verify everything works
2. **Start training** with `bash scripts/train_etram.sh`
3. **Monitor training** with `tail -f` on the log file
4. **Evaluate** when training completes
5. **Visualize** predictions in the evaluation folder

---

## Full Command Reference

### Training Commands

```bash
# Standard training
CUDA_VISIBLE_DEVICES=1 python train_etram.py --data_root DATA_ROOT --save_dir checkpoints/exp1

# With object masking
CUDA_VISIBLE_DEVICES=1 python train_etram.py --data_root DATA_ROOT --use_obj_mask

# Custom hyperparameters
CUDA_VISIBLE_DEVICES=1 python train_etram.py \
    --pre_seq 20 --aft_seq 20 --batch_size 4 --epochs 200 --lr 5e-4
```

### Evaluation Commands

```bash
# Evaluate on test set
CUDA_VISIBLE_DEVICES=1 python evaluate_etram.py \
    --checkpoint checkpoints/exp1/best_model_f1.pth \
    --visualize --num_vis_samples 20

# Quick evaluation without visualization
CUDA_VISIBLE_DEVICES=1 python evaluate_etram.py \
    --checkpoint checkpoints/exp1/best_model_f1.pth \
    --no-visualize
```

---

## Contact & Support

- Check `README_ETRAM.md` for detailed documentation
- Review `test_setup.py` output for setup issues
- Examine `args.json` in checkpoint folder for training configuration

Happy Training! 🚀
