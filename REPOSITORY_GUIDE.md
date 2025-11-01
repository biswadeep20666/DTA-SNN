# DTA-SNN Repository Guide - Explained in Simple Words

## What is this repository about?

This repository implements **DTA (Dual Temporal-channel-wise Attention)** for **Spiking Neural Networks (SNNs)**. It's a research project published at WACV 2025 conference.

### In Very Simple Terms:

Think of it like this:
- **Regular neural networks** work with continuous values (like 0.5, 0.7, etc.)
- **Spiking neural networks** work more like the human brain - they use discrete "spikes" (on/off signals) over time
- **DTA is a special attention mechanism** that helps SNNs focus on important information in two ways:
  1. **Temporal attention** (T_XA) - focuses on important time moments
  2. **Spatial attention** (T_NA) - focuses on important spatial features

---

## Repository Structure

```
DTA-SNN/
├── main.py                  # Main training script - this is where you start
├── models/                  # All the neural network architectures
│   ├── DTA.py              # Main DTA module (combines T_XA and T_NA)
│   ├── TXA.py              # Temporal-Channel cross Attention module
│   ├── TNA.py              # Temporal-Neuron Attention module
│   ├── DTA_SNN.py          # Complete SNN model with DTA
│   └── layers.py           # Building blocks (LIF neurons, convolutions, etc.)
├── data/                    # Data loading and preprocessing
│   ├── loaders.py          # CIFAR-10/100 dataset loaders
│   ├── datasets.py         # ImageNet dataset loader
│   └── augmentations.py    # Data augmentation functions
├── scripts/                 # Easy-to-use training scripts
│   ├── run_cifar10.sh      # Train on CIFAR-10
│   ├── run_cifar100.sh     # Train on CIFAR-100
│   ├── run_imgnet.sh       # Train on ImageNet
│   └── run_dvs_cifar10.sh  # Train on DVS CIFAR-10
├── utils/                   # Helper functions
│   └── utils.py            # Training utilities
├── requirements.txt         # Python packages needed
├── Dockerfile              # Docker container setup
└── README.md               # Basic documentation

```

---

## Key Components Explained

### 1. **DTA Module** (`models/DTA.py`)
This is the heart of the innovation. It combines two attention mechanisms:
- **T_NA (Temporal-Neuron Attention)**: Looks at spatial patterns across time
- **T_XA (Temporal-Channel cross Attention)**: Looks at temporal and channel patterns
- They work together to highlight important spikes and suppress noise

### 2. **T_XA Module** (`models/TXA.py`)
- **Temporal attention**: Focuses on important time steps
- **Channel attention**: Focuses on important feature channels
- Uses 1D convolutions and learnable scaling parameters
- Output: Enhanced feature representation

### 3. **T_NA Module** (`models/TNA.py`)
- **Local-Temporal-Channel Attention (LTCA)**: Uses depthwise convolutions
- **Global-Temporal-Channel Attention (GTCA)**: Uses global average pooling + MLP
- Combines local and global spatial information

### 4. **DTA_SNN Model** (`models/DTA_SNN.py`)
- Complete spiking neural network architectures
- Two versions: **DTA_SNN_18** (ResNet-18 style) and **DTA_SNN_34** (ResNet-34 style)
- Uses **LIF (Leaky Integrate-and-Fire) neurons** - these mimic biological neurons
- DTA module is inserted after the first convolution layer

### 5. **LIF Neurons** (`models/layers.py`)
- Simulates biological neuron behavior
- Accumulates input over time (integrate)
- Fires a spike when threshold is reached
- Has memory decay (leaky)

---

## How to Use This Repository

### Step 1: Set Up Environment

**Option A: Using Docker (Recommended)**
```bash
# Build and run the Docker container
docker-compose -f docker-compose-gpu.yaml up -d
```

**Option B: Manual Setup**
```bash
# Install Python 3.10
# Install PyTorch 1.13.1 with CUDA 11.6
# Install dependencies
pip install -r requirements.txt
```

Required packages:
- PyTorch 1.13.1
- spikingjelly (SNN library)
- timm (vision models)
- opencv-python-headless
- albumentations (data augmentation)
- numpy, scipy

### Step 2: Prepare Datasets

Create a `dataset/` folder with this structure:

```
dataset/
├── CIFAR/              # For CIFAR-10/100 (auto-downloaded)
├── DVS_CIFAR10/        # For DVS CIFAR-10
│   └── frames_number_10_split_by_number/
└── ImageNet/           # For ImageNet
    ├── train/
    └── val/
```

### Step 3: Train the Model

**Easy way - Use the provided scripts:**

```bash
# Train on CIFAR-10
bash ./scripts/run_cifar10.sh

# Train on CIFAR-100
bash ./scripts/run_cifar100.sh

# Train on ImageNet
bash ./scripts/run_imgnet.sh

# Train on DVS CIFAR-10
bash ./scripts/run_dvs_cifar10.sh
```

**Manual way - Customize your training:**

```bash
python main.py \
    --DS cifar10 \              # Dataset: cifar10, cifar100, imgnet, dvs_cifar10
    --epochs 250 \              # Number of training epochs
    --batch_size 64 \           # Batch size
    --learning_rate 0.1 \       # Initial learning rate
    --time_step 6 \             # Number of time steps for SNN simulation
    --seed 42 \                 # Random seed for reproducibility
    --DTA_ON True \             # Use DTA attention (True/False)
    --weight_decay 5e-5         # Weight decay for regularization
```

### Step 4: What Happens During Training

1. **Load dataset** - CIFAR-10, CIFAR-100, ImageNet, or DVS CIFAR-10
2. **Create model** - DTA_SNN_18 or DTA_SNN_34 with or without DTA
3. **Training loop**:
   - Forward pass: Images → Repeated T times → SNN processing → Spikes → Mean → Predictions
   - Loss calculation: CrossEntropyLoss
   - Backward pass: Update weights
   - Data augmentation: Mixup, Cutmix, random crops, flips
4. **Validation** - Test accuracy after each epoch
5. **Save best model** - Saves when validation accuracy improves

### Step 5: Output

The training will:
- Print training loss and accuracy for each epoch
- Print test accuracy after each epoch
- Save the best model as a `.pth.tar` file with format:
  ```
  CIFAR10-S42-B64-T6-E250-LR0.1.pth.tar
  ```
  (Dataset-Seed-BatchSize-TimeSteps-Epochs-LearningRate)

---

## Key Concepts Explained

### What are Spiking Neural Networks?
- Brain-inspired neural networks
- Use binary spikes (0 or 1) instead of continuous values
- Process information over time (temporal dimension)
- More energy-efficient than regular neural networks

### What is the Time Step?
- SNNs process data over multiple time steps (default: 6)
- Each image is repeated T times and processed sequentially
- Neurons accumulate information and fire spikes over time
- Final output is the average of all time steps

### What is Attention?
- A mechanism to focus on important information
- Like highlighting important words in a text
- DTA has two types:
  1. **Temporal attention**: Which time moments are important?
  2. **Channel attention**: Which features are important?

### What is the purpose of DTA?
- Regular SNNs can be noisy and less accurate
- DTA helps filter out noise and focus on important spikes
- Improves accuracy while maintaining SNN benefits
- Works better than standard attention mechanisms for SNNs

---

## Training Parameters Explained

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `--DS` | Dataset to use | `cifar10`, `cifar100`, `imgnet`, `dvs_cifar10` |
| `--epochs` | How many times to go through the entire dataset | 250 |
| `--batch_size` | Number of images processed together | 64 |
| `--learning_rate` | How fast the model learns | 0.1 |
| `--time_step` | Number of time steps for SNN simulation | 6 |
| `--DTA_ON` | Enable/disable DTA attention | True |
| `--weight_decay` | Regularization to prevent overfitting | 5e-5 |
| `--seed` | Random seed for reproducibility | 42 |
| `--mixup` | Mixup augmentation strength | 0.5 |
| `--cutmix_prob` | Probability of using CutMix | 0.5 |

---

## Common Use Cases

### 1. **Train a basic model on CIFAR-10**
```bash
bash ./scripts/run_cifar10.sh
```
This is the easiest way to start and verify everything works.

### 2. **Compare with/without DTA**
```bash
# With DTA
python main.py --DS cifar10 --DTA_ON True --epochs 250

# Without DTA (baseline)
python main.py --DS cifar10 --DTA_ON False --epochs 250
```

### 3. **Train on your own dataset**
You would need to:
1. Create a data loader similar to `data/loaders.py`
2. Add dataset option in `main.py`
3. Adjust model parameters if needed

### 4. **Resume training**
Currently, the code trains from scratch. To resume, you'd need to:
1. Load the saved model weights
2. Set `--start_epoch` to the last completed epoch

---

## Performance Expectations

Based on the paper (WACV 2025):

| Dataset | Time Steps | Accuracy |
|---------|-----------|----------|
| CIFAR-10 | 6 | ~95%+ |
| CIFAR-100 | 6 | ~75%+ |
| ImageNet | 6 | ~70%+ |
| DVS CIFAR-10 | 10 | ~80%+ |

*Note: Exact numbers depend on hyperparameters and training setup*

---

## Troubleshooting

### Problem: Out of memory
**Solution**: Reduce `--batch_size` (try 32 or 16)

### Problem: Dataset not found
**Solution**: Check the `dataset/` folder structure matches the expected format

### Problem: Training is too slow
**Solution**: 
- Use GPU if available
- Reduce `--time_step` (try 4 instead of 6)
- Use smaller dataset (CIFAR-10 instead of ImageNet)

### Problem: Low accuracy
**Solution**:
- Train for more epochs
- Check if DTA is enabled (`--DTA_ON True`)
- Verify data augmentation is working
- Try different learning rate

---

## How to Experiment

### 1. **Change the architecture**
Edit `models/DTA_SNN.py`:
- Modify layer depths
- Change number of channels
- Adjust attention mechanisms

### 2. **Try different attention**
Edit `models/DTA.py`:
- Change how T_NA and T_XA are combined
- Add new attention mechanisms

### 3. **Adjust SNN behavior**
Edit `models/layers.py`:
- Change LIF neuron parameters (threshold, tau)
- Modify spike generation

### 4. **Add new datasets**
Edit `main.py`:
- Add new dataset loading logic
- Create new data loader in `data/`

---

## Understanding the Code Flow

### Training Flow:
```
main.py (start here)
    ↓
Load dataset (data/loaders.py or data/datasets.py)
    ↓
Create model (models/DTA_SNN.py)
    ↓
Training loop:
    ↓
    Images → Repeat T times → [Batch, Time, Channel, Height, Width]
    ↓
    First Conv + LIF neurons → Generate spikes
    ↓
    DTA module (if enabled) → Enhance important spikes
        ↓
        T_NA → Spatial attention
        T_XA → Temporal & Channel attention
    ↓
    ResNet layers → Feature extraction
    ↓
    Average over time → Final prediction
    ↓
    Loss → Backprop → Update weights
    ↓
    Validation → Save best model
```

### Model Architecture:
```
Input Image [B, C, H, W]
    ↓
Repeat T times → [B, T, C, H, W]
    ↓
Conv1 + LIF → Spikes [B, T, 64, H/2, W/2]
    ↓
DTA (if enabled):
    ├─ T_NA (spatial attention)
    └─ T_XA (temporal & channel attention)
    ↓
Layer 0: ResNet blocks → [B, T, 64, H/2, W/2]
    ↓
Layer 1: ResNet blocks → [B, T, 128, H/4, W/4]
    ↓
Layer 2: ResNet blocks → [B, T, 256, H/8, W/8]
    ↓
Layer 3: ResNet blocks → [B, T, 512, H/16, W/16]
    ↓
Average Pooling → [B, T, 512, 1, 1]
    ↓
Fully Connected → [B, T, num_classes]
    ↓
Mean over time → [B, num_classes]
    ↓
Output: Class predictions
```

---

## Citation

If you use this code, cite the paper:
```bibtex
@InProceedings{Kim_2025_WACV,
    author    = {Kim, Minje and Kim, Minjun and Yang, Xu},
    title     = {DTA: Dual Temporal-Channel-Wise Attention for Spiking Neural Networks},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {9682-9692}
}
```

---

## Summary

**In one sentence**: This repository trains Spiking Neural Networks with a special attention mechanism (DTA) that helps the network focus on important spikes, making it more accurate for image classification tasks.

**Why it matters**: SNNs are brain-inspired and energy-efficient, but they're harder to train. DTA makes them more accurate and competitive with regular neural networks.

**How to get started**: 
1. Install requirements
2. Run `bash ./scripts/run_cifar10.sh`
3. Watch it train!

**Key innovation**: DTA combines temporal and spatial attention specifically designed for SNNs, unlike regular attention mechanisms designed for standard neural networks.

---

## Questions to Ask Yourself When Using This Repo

1. **What dataset do I want to use?** → Choose CIFAR-10 for quick experiments, ImageNet for serious training
2. **Do I have a GPU?** → Essential for reasonable training times
3. **What's my goal?** → Reproduce paper results? Experiment with SNNs? Compare with/without DTA?
4. **How much time do I have?** → CIFAR training takes hours, ImageNet takes days
5. **Do I need to modify the code?** → For basic usage, no. For research, maybe.

---

## Additional Resources

- **Paper**: [WACV 2025 Paper](https://openaccess.thecvf.com/content/WACV2025/papers/Kim_DTA_Dual_Temporal-Channel-Wise_Attention_for_Spiking_Neural_Networks_WACV_2025_paper.pdf)
- **SpikingJelly**: [GitHub](https://github.com/fangwei123456/spikingjelly) - SNN library used
- **ResNet**: Original architecture this is based on
- **Attention Mechanisms**: Background on attention in neural networks

---

**Need help?** Look at:
1. The training scripts in `scripts/` - they show working examples
2. The README.md - basic instructions
3. The paper - detailed methodology and results
4. The code comments - inline explanations

**Remember**: You don't need to create PRs or make code changes to use this repository. Just run the training scripts and experiment with parameters!
