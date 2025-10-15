# DTA-SNN Architecture Visualization

## Overall System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                               │
│  "I want to train an SNN on CIFAR-10 with DTA attention"       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RUN TRAINING SCRIPT                           │
│         bash ./scripts/run_cifar10.sh                           │
│                                                                 │
│  Or: python main.py --DS cifar10 --DTA_ON True --epochs 250    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MAIN.PY - TRAINING LOOP                        │
│  1. Load Dataset (CIFAR-10/100, ImageNet, DVS)                 │
│  2. Create Model (DTA_SNN_18 or DTA_SNN_34)                    │
│  3. Train for 250 epochs                                       │
│  4. Save best model                                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MODEL ARCHITECTURE                            │
│                  (DTA_SNN_18/34)                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Model Architecture

```
INPUT IMAGE
   [32x32x3]
      │
      ▼
┌─────────────────────────────────────┐
│   REPEAT T TIMES (e.g., T=6)        │
│   [Batch, Time=6, 3, 32, 32]        │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│   CONV1 (7x7) + BatchNorm            │
│   Output: [B, T, 64, 16, 16]        │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│   LIF NEURONS (Generate Spikes)     │
│   Membrane potential → Spikes       │
│   Output: [B, T, 64, 16, 16]        │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│        DTA MODULE (if enabled)      │
│   ┌─────────────┬─────────────┐     │
│   │    T_NA     │    T_XA     │     │
│   │  (Spatial)  │ (Temporal+  │     │
│   │  Attention  │  Channel)   │     │
│   └──────┬──────┴──────┬──────┘     │
│          │             │            │
│          └─────┬───────┘            │
│                ▼                    │
│         Combined Attention          │
│         ↓                           │
│   Enhanced Spikes                   │
│   Output: [B, T, 64, 16, 16]        │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│   LAYER 0: ResNet Blocks            │
│   Output: [B, T, 64, 16, 16]        │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│   LAYER 1: ResNet Blocks            │
│   (stride=1, 128 channels)          │
│   Output: [B, T, 128, 16, 16]       │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│   LAYER 2: ResNet Blocks            │
│   (stride=2, 256 channels)          │
│   Output: [B, T, 256, 8, 8]         │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│   LAYER 3: ResNet Blocks            │
│   (stride=2, 512 channels)          │
│   Output: [B, T, 512, 4, 4]         │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│   GLOBAL AVERAGE POOLING            │
│   Output: [B, T, 512, 1, 1]         │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│   FULLY CONNECTED LAYER             │
│   Output: [B, T, num_classes]       │
└─────────┬───────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│   MEAN OVER TIME                    │
│   Output: [B, num_classes]          │
└─────────┬───────────────────────────┘
          │
          ▼
    PREDICTIONS
   [10 classes for CIFAR-10]
```

---

## DTA Module Deep Dive

```
INPUT SPIKES: [Batch, Time, Channels, Height, Width]
                   [B,    T,     C,       H,      W]
      │
      ├─────────────────┬─────────────────┐
      │                 │                 │
      ▼                 ▼                 ▼
┌──────────┐      ┌──────────┐      ┌──────────┐
│ ORIGINAL │      │   T_NA   │      │   T_XA   │
│  SPIKES  │      │  PATH    │      │   PATH   │
└──────────┘      └────┬─────┘      └────┬─────┘
      │                │                  │
      │                ▼                  ▼
      │      ┌──────────────────┐  ┌──────────────────┐
      │      │ Reshape to       │  │ Keep 5D shape    │
      │      │ [B, T*C, H, W]   │  │ [B, T, C, H, W]  │
      │      └────┬─────────────┘  └────┬─────────────┘
      │           │                     │
      │           ▼                     ▼
      │      ┌──────────────────┐  ┌──────────────────┐
      │      │ T_NA Module:     │  │ T_XA Module:     │
      │      │ - Encoding       │  │ - Temporal Conv  │
      │      │ - TCA (Local+    │  │ - Channel Conv   │
      │      │   Global Attn)   │  │ - Sigmoid        │
      │      │ - Decoding       │  │ - Scaling        │
      │      └────┬─────────────┘  └────┬─────────────┘
      │           │                     │
      │           ▼                     ▼
      │      ┌──────────────────┐  ┌──────────────────┐
      │      │ Reshape back to  │  │ Attention Maps:  │
      │      │ [B, T, C, H, W]  │  │ - Temporal attn  │
      │      │                  │  │ - Channel attn   │
      │      └────┬─────────────┘  └────┬─────────────┘
      │           │                     │
      │           ▼                     ▼
      │      [Spatial Attention]   [Temporal+Channel]
      │           │                     │
      │           └──────┬──────────────┘
      │                  │
      │                  ▼
      │         ┌─────────────────┐
      │         │  MULTIPLY T_NA  │
      │         │     × T_XA      │
      │         └────┬────────────┘
      │              │
      │              ▼
      │         ┌─────────────────┐
      │         │    SIGMOID      │
      │         └────┬────────────┘
      │              │
      │              ▼
      │      [Combined Attention]
      │              │
      └──────┬───────┘
             │
             ▼
       ┌─────────────────┐
       │  MULTIPLY WITH  │
       │ ORIGINAL SPIKES │
       └────┬────────────┘
            │
            ▼
      ENHANCED SPIKES
   [B, T, C, H, W]
```

---

## T_XA Module (Temporal-Channel Cross Attention)

```
INPUT: [B, T, C, H, W]
   │
   ├───────────────┬───────────────┐
   │               │               │
   ▼               ▼               ▼
TEMPORAL        CHANNEL        ORIGINAL
BRANCH          BRANCH          INPUT
   │               │               │
   ▼               ▼               │
┌─────────┐   ┌─────────┐         │
│ Mean    │   │ Mean    │         │
│ spatial │   │ spatial │         │
│ dims    │   │ dims    │         │
└────┬────┘   └────┬────┘         │
     │             │               │
     ▼             ▼               │
[B,T,C]      [B,T,C]              │
     │             │               │
     ▼             ▼               │
┌─────────┐   ┌─────────┐         │
│ Conv1d  │   │ Permute │         │
│ over T  │   │ to      │         │
│         │   │ [B,C,T] │         │
└────┬────┘   └────┬────┘         │
     │             │               │
     ▼             ▼               │
┌─────────┐   ┌─────────┐         │
│         │   │ Conv1d  │         │
│         │   │ over C  │         │
│         │   │         │         │
└────┬────┘   └────┬────┘         │
     │             │               │
     ▼             ▼               │
┌─────────┐   ┌─────────┐         │
│ Sigmoid │   │ Permute │         │
│         │   │ back    │         │
└────┬────┘   └────┬────┘         │
     │             │               │
     ▼             ▼               │
[Temporal]    [Channel]           │
[Attention]   [Attention]         │
     │             │               │
     ├─────┬───────┤               │
     ▼     ▼       ▼               │
┌──────┐ ┌───┐ ┌──────┐           │
│scale_t│×│   │×│scale_c│          │
└───┬──┘ └─┬─┘ └───┬──┘           │
    │      │       │               │
    └──┬───┴───┬───┘               │
       ▼       ▼                   │
  ┌─────────────────┐              │
  │ Add to original │              │
  │ (Residual)      │              │
  └────┬─────┬──────┘              │
       │     │                     │
       ▼     ▼                     │
  [attn_t] [attn_c]                │
       │     │                     │
       └──┬──┘                     │
          ▼                        │
    ┌──────────┐                   │
    │ MULTIPLY │                   │
    └────┬─────┘                   │
         │                         │
         ▼                         │
    OUTPUT: Enhanced features
    [B, T, C, H, W]
```

---

## T_NA Module (Temporal-Neuron Attention)

```
INPUT: [B, T*C, H, W]
   │
   ├──────────────────┐
   │                  │
   ▼                  ▼
ATTENTION         SHORTCUT
PATH              (IDENTITY)
   │                  │
   ▼                  │
┌─────────┐           │
│ Encoding│           │
│ Conv2d  │           │
│  + GELU │           │
└────┬────┘           │
     │                │
     ▼                │
┌─────────┐           │
│   TCA   │           │
│ Module  │           │
└────┬────┘           │
     │                │
     ├────────┐       │
     │        │       │
     ▼        ▼       │
┌─────────┐ ┌─────┐  │
│  LTCA   │ │ GAP │  │
│ (Local) │ │     │  │
└────┬────┘ └──┬──┘  │
     │         │      │
     │         ▼      │
     │      ┌─────┐   │
     │      │ MLP │   │
     │      └──┬──┘   │
     │         │      │
     │         ▼      │
     │    [GTCA]      │
     │    (Global)    │
     │         │      │
     └────┬────┘      │
          │           │
          ▼           │
    ┌──────────┐      │
    │ MULTIPLY │      │
    │  × Input │      │
    └────┬─────┘      │
         │            │
         ▼            │
    ┌──────────┐      │
    │ Decoding │      │
    │  Conv2d  │      │
    └────┬─────┘      │
         │            │
         └────┬───────┘
              │
              ▼
          ┌───────┐
          │  ADD  │
          │   +   │
          └───┬───┘
              │
              ▼
         OUTPUT
    [B, T*C, H, W]
```

---

## LIF Neuron Operation

```
TIME STEP 1    TIME STEP 2    TIME STEP 3    ...    TIME STEP T
    │              │              │                      │
    ▼              ▼              ▼                      ▼
┌────────┐     ┌────────┐     ┌────────┐           ┌────────┐
│ Input  │     │ Input  │     │ Input  │           │ Input  │
│  x[1]  │     │  x[2]  │     │  x[3]  │           │  x[T]  │
└───┬────┘     └───┬────┘     └───┬────┘           └───┬────┘
    │              │              │                      │
    ▼              ▼              ▼                      ▼
┌────────────────────────────────────────────────────────────┐
│              MEMBRANE POTENTIAL (mem)                      │
│                                                            │
│  mem[t] = mem[t-1] * tau + x[t]                           │
│         └─ decay        └─ new input                       │
│                                                            │
│  tau = 0.5 (leak parameter)                               │
└────────────┬───────────────────────────────────────────────┘
             │
             ▼
        ┌─────────┐
        │ mem ≥   │───YES──┐
        │ thresh? │        │
        └────┬────┘        │
             │             │
             NO            ▼
             │      ┌────────────┐
             │      │ FIRE SPIKE │
             │      │  (output=1)│
             │      └──────┬─────┘
             │             │
             │             ▼
             │      ┌────────────┐
             │      │ RESET mem  │
             │      │ mem = 0    │
             │      └──────┬─────┘
             │             │
             ▼             ▼
        ┌─────────────────────┐
        │ NO SPIKE (output=0) │
        │ Keep mem            │
        └──────────┬──────────┘
                   │
                   ▼
            NEXT TIME STEP

SUMMARY:
- Accumulate input over time (integrate)
- Spike when threshold reached (fire)
- Reset after spike (refractory)
- Membrane decays over time (leaky)
```

---

## Training Data Flow

```
EPOCH 1                    EPOCH 2               ...    EPOCH 250
   │                          │                            │
   ▼                          ▼                            ▼
┌──────────────────────────────────────────────────────────────┐
│                  LOAD BATCH OF IMAGES                        │
│           [Batch=64, 3 channels, 32×32 pixels]               │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                   DATA AUGMENTATION                          │
│  • Random crop with padding                                  │
│  • Random horizontal flip                                    │
│  • Mixup (mix two images with random weight)                 │
│  • CutMix (cut patch from one image, paste to another)       │
│  • Normalization                                            │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    FORWARD PASS                              │
│  Images → Model → Predictions                                │
│  • Repeat T times (temporal dimension)                       │
│  • Process through SNN layers                                │
│  • Apply DTA attention                                       │
│  • Extract features                                          │
│  • Average over time                                         │
│  • Output class probabilities                                │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  COMPUTE LOSS                                │
│  • CrossEntropyLoss between predictions and labels           │
│  • Special handling for Mixup/CutMix                        │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  BACKWARD PASS                               │
│  • Compute gradients                                         │
│  • Update weights using optimizer (SGD)                      │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│               VALIDATION (after each epoch)                  │
│  • Test on validation set                                    │
│  • Compute accuracy                                          │
│  • Save model if best accuracy so far                        │
└──────────────────────────────────────────────────────────────┘
```

---

## File Dependencies

```
main.py
   │
   ├──> models/DTA_SNN.py
   │       │
   │       ├──> models/DTA.py
   │       │       │
   │       │       ├──> models/TXA.py
   │       │       └──> models/TNA.py
   │       │
   │       └──> models/layers.py
   │               (LIFSpike, BasicBlock_MS, etc.)
   │
   ├──> data/datasets.py
   │       │
   │       └──> data/loaders.py
   │
   ├──> data/augmentations.py
   │       (Mixup, CutMix functions)
   │
   └──> utils/utils.py
           (seed_all, logging, etc.)

EXTERNAL DEPENDENCIES (requirements.txt):
   ├──> spikingjelly (SNN utilities)
   ├──> timm (vision models)
   ├──> opencv-python-headless
   ├──> albumentations (augmentation)
   ├──> numpy, scipy
   └──> PyTorch 1.13.1
```

---

## Summary Diagram

```
┌────────────────────────────────────────────────────────────┐
│                     DTA-SNN SYSTEM                         │
│                                                            │
│  INPUT: Images (32×32 or 224×224)                         │
│    ↓                                                       │
│  PREPROCESSING: Augmentation, Normalization                │
│    ↓                                                       │
│  SNN ENCODING: Repeat T times                              │
│    ↓                                                       │
│  CONV1 + LIF: Convert to spikes                           │
│    ↓                                                       │
│  DTA MODULE: Enhance important spikes                      │
│    ├─ T_NA: Spatial attention                             │
│    └─ T_XA: Temporal + Channel attention                   │
│    ↓                                                       │
│  RESNET LAYERS: Feature extraction                         │
│    ↓                                                       │
│  GLOBAL POOLING: Spatial reduction                         │
│    ↓                                                       │
│  FULLY CONNECTED: Classification                           │
│    ↓                                                       │
│  TEMPORAL MEAN: Average over time steps                    │
│    ↓                                                       │
│  OUTPUT: Class predictions                                 │
│                                                            │
│  TRAINING: Update weights via backpropagation              │
│  VALIDATION: Test accuracy, save best model                │
└────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **DTA = T_NA + T_XA**: Two complementary attention mechanisms
2. **SNN = Time-based processing**: Multiple time steps (T=6 typically)
3. **LIF neurons**: Brain-like spiking behavior
4. **ResNet backbone**: Standard architecture adapted for SNNs
5. **Training**: Standard supervised learning with special augmentations
6. **Output**: Trained model saved as `.pth.tar` file

For more details, see:
- `QUICK_START.md` - Quick reference guide
- `REPOSITORY_GUIDE.md` - Comprehensive explanation
- `README.md` - Basic usage instructions
