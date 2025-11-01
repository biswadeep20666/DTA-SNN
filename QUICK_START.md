# Quick Start Guide for DTA-SNN

## What is this?
A research project that trains **Spiking Neural Networks** (brain-like AI) with a special **attention mechanism** (DTA) for image classification.

## In 3 Steps:

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Train
```bash
bash ./scripts/run_cifar10.sh
```

### 3. Done!
The model will train and save the best weights automatically.

---

## What You Need to Know

### Files You'll Touch:
- **`scripts/run_cifar10.sh`** - Run this to start training
- **`main.py`** - The main training script (customize if needed)
- **`requirements.txt`** - Python packages to install

### Files You Don't Need to Touch (But Are Important):
- **`models/DTA.py`** - The DTA attention mechanism
- **`models/TXA.py`** - Temporal-Channel attention
- **`models/TNA.py`** - Temporal-Neuron attention
- **`models/DTA_SNN.py`** - The complete neural network
- **`models/layers.py`** - Building blocks (neurons, layers)

---

## Simple Explanations

### What is a Spiking Neural Network (SNN)?
Think of regular neural networks like a smooth river flowing continuously. SNNs are more like raindrops - discrete bursts of activity, just like your brain's neurons!

### What is DTA?
**D**ual **T**emporal-channel-wise **A**ttention = A smart filter that helps the network focus on:
- **When** important things happen (temporal)
- **What** features matter (channel)

### Why use DTA?
- Makes SNNs more accurate
- Filters out noise from spikes
- Better than regular attention for SNNs

---

## Common Commands

### Train on CIFAR-10 (easiest, fastest)
```bash
bash ./scripts/run_cifar10.sh
```

### Train on CIFAR-100 (harder, 100 classes)
```bash
bash ./scripts/run_cifar100.sh
```

### Train on ImageNet (hardest, needs lots of GPU memory)
```bash
bash ./scripts/run_imgnet.sh
```

### Train on DVS CIFAR-10 (neuromorphic camera data)
```bash
bash ./scripts/run_dvs_cifar10.sh
```

### Custom training
```bash
python main.py --DS cifar10 --epochs 250 --batch_size 64 --learning_rate 0.1 --time_step 6
```

---

## Key Parameters

| What | Flag | Example |
|------|------|---------|
| Dataset | `--DS` | `cifar10`, `cifar100`, `imgnet` |
| Training rounds | `--epochs` | `250` |
| Images at once | `--batch_size` | `64` (reduce if out of memory) |
| Learning speed | `--learning_rate` | `0.1` |
| Time steps | `--time_step` | `6` (how many time steps the SNN runs) |
| Use DTA? | `--DTA_ON` | `True` or `False` |

---

## What Happens When You Train?

1. **Load images** (CIFAR-10 by default)
2. **Create the brain-like network** (SNN with DTA)
3. **Train for many rounds** (epochs)
   - Show images to the network
   - Network makes predictions
   - Learn from mistakes
   - Get better over time
4. **Save the best version** (when accuracy is highest)
5. **Done!** You have a trained model

---

## Expected Results

Training on CIFAR-10 with default settings:
- **Time**: 2-4 hours on a good GPU
- **Accuracy**: Around 95%+
- **Output**: A `.pth.tar` file with the trained model

---

## Troubleshooting

### "Out of memory"
→ Reduce batch size: `--batch_size 32` or `--batch_size 16`

### "Dataset not found"
→ The code will auto-download CIFAR-10/100. For ImageNet, you need to download it manually.

### "Training is slow"
→ Make sure you're using a GPU. Check with: `nvidia-smi`

### "Accuracy is low"
→ Make sure DTA is enabled: `--DTA_ON True`

---

## File Output

After training, you'll get a file like:
```
CIFAR10-S42-B64-T6-E250-LR0.1.pth.tar
```

This means:
- **CIFAR10**: Trained on CIFAR-10
- **S42**: Random seed 42
- **B64**: Batch size 64
- **T6**: 6 time steps
- **E250**: 250 epochs
- **LR0.1**: Learning rate 0.1

---

## Next Steps

1. **Read the full guide**: Check `REPOSITORY_GUIDE.md` for detailed explanations
2. **Read the paper**: [WACV 2025 Paper](https://openaccess.thecvf.com/content/WACV2025/papers/Kim_DTA_Dual_Temporal-Channel-Wise_Attention_for_Spiking_Neural_Networks_WACV_2025_paper.pdf)
3. **Experiment**: Try different parameters, datasets, or architectures
4. **Modify**: Change the attention mechanism or network architecture

---

## One More Thing

**You don't need to create pull requests (PRs) to use this code!**

PRs are for contributing changes back to the repository. If you just want to:
- Train models
- Experiment with parameters  
- Test on your data
- Learn about SNNs

Just clone the repo and use it directly. No PRs needed! 😊

---

## Questions?

- Check `REPOSITORY_GUIDE.md` for detailed explanations
- Look at the code comments in `main.py`
- Read the training scripts in `scripts/`
- Review the paper for methodology

**Happy training! 🚀**
