# Documentation Index - DTA-SNN Repository Explained

## 📚 Welcome!

This repository has been thoroughly documented to help you understand and use it **without needing to create pull requests**. Below is your guide to all the documentation.

---

## 🎯 Start Here Based on Your Needs

### "I'm completely new and just want the basics"
👉 **Read: [NO_PR_GUIDE.md](NO_PR_GUIDE.md)**
- Answers your main question about using the repo without PRs
- Explains what PRs are (and why you don't need them)
- Simple download and run instructions

### "I want to get started quickly"
👉 **Read: [QUICK_START.md](QUICK_START.md)**
- 3-step quickstart
- Common commands
- Key parameters explained
- Troubleshooting tips

### "I want to understand everything in detail"
👉 **Read: [REPOSITORY_GUIDE.md](REPOSITORY_GUIDE.md)**
- Complete explanation of DTA-SNN
- Repository structure
- Key concepts explained (SNNs, attention, LIF neurons)
- Training workflow
- How to experiment and modify

### "I want to see visual diagrams of the architecture"
👉 **Read: [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md)**
- System flow diagrams
- Model architecture visualizations
- DTA module deep dive with ASCII art
- Training data flow
- File dependency maps

### "I just want the official basics"
👉 **Read: [README.md](README.md)**
- Official repository documentation
- Requirements
- Training instructions
- Dataset structure
- Citation

---

## 📖 Document Summaries

### 1. NO_PR_GUIDE.md
**What it covers:**
- What are Pull Requests and when you need them
- How to download/clone the repository
- How to use the code locally without PRs
- Multiple usage scenarios (training, experimenting, modifying)
- Git basics without pushing
- FAQ about PRs

**Best for:** Beginners who are worried about accidentally creating PRs

**Key takeaway:** You can use this repository completely locally. PRs are only for contributing changes back to the original repository, which is optional!

---

### 2. QUICK_START.md
**What it covers:**
- 3-step quickstart (Install, Train, Done)
- Simple explanations (SNN, DTA, Time Steps)
- Common training commands for different datasets
- Key parameter reference table
- What happens during training
- Expected results and timing
- Troubleshooting common issues
- File output explanation

**Best for:** Getting up and running quickly

**Key takeaway:** `bash ./scripts/run_cifar10.sh` is all you need to start!

---

### 3. REPOSITORY_GUIDE.md
**What it covers:**
- What is DTA-SNN (in simple terms)
- Complete repository structure
- Detailed component explanations:
  - DTA Module (combines T_NA and T_XA)
  - T_XA (Temporal-Channel cross Attention)
  - T_NA (Temporal-Neuron Attention)
  - LIF Neurons (Leaky Integrate-and-Fire)
  - DTA_SNN Model architecture
- Step-by-step usage instructions
- Training process explained
- Key concepts (SNNs, Attention, Time Steps)
- All training parameters explained
- Common use cases
- How to experiment and modify
- Performance expectations
- Troubleshooting
- Understanding code flow

**Best for:** Deep understanding of the entire system

**Key takeaway:** Comprehensive guide covering everything from concepts to implementation

---

### 4. ARCHITECTURE_DIAGRAM.md
**What it covers:**
- Overall system flow (visual)
- Detailed model architecture (layer by layer)
- DTA module breakdown with diagrams
- T_XA module visualization (temporal + channel paths)
- T_NA module visualization (local + global attention)
- LIF neuron operation (time step by time step)
- Training data flow
- File dependencies map
- Summary diagrams

**Best for:** Visual learners who want to see the architecture

**Key takeaway:** ASCII art diagrams showing how data flows through the entire system

---

### 5. README.md (Official)
**What it covers:**
- Project title and paper link
- Requirements (Python 3.10, PyTorch 1.13.1, CUDA 11.6)
- Training commands
- Dataset structure
- Citation information

**Best for:** Official reference

**Key takeaway:** The original, concise documentation from the authors

---

## 🗺️ Recommended Reading Path

### Path 1: Quick Start (15 minutes)
1. **NO_PR_GUIDE.md** - Understand you don't need PRs (5 min)
2. **QUICK_START.md** - Get running (10 min)
3. Start training! 🚀

### Path 2: Comprehensive Understanding (1-2 hours)
1. **NO_PR_GUIDE.md** - Setup understanding (10 min)
2. **QUICK_START.md** - Basic concepts (15 min)
3. **REPOSITORY_GUIDE.md** - Deep dive (45 min)
4. **ARCHITECTURE_DIAGRAM.md** - Visual understanding (30 min)
5. Experiment with the code! 🔬

### Path 3: Just the Essentials (5 minutes)
1. **QUICK_START.md** - Read "In 3 Steps" section
2. Run: `bash ./scripts/run_cifar10.sh`
3. Done! ✅

---

## 🔍 Finding Specific Information

### "How do I install dependencies?"
- **QUICK_START.md** - Step 1
- **NO_PR_GUIDE.md** - Step 2
- **README.md** - Requirements section

### "What datasets can I use?"
- **QUICK_START.md** - Common Commands section
- **REPOSITORY_GUIDE.md** - Training Parameters table
- **README.md** - Dataset Structure section

### "What is DTA and how does it work?"
- **QUICK_START.md** - "What is DTA?" section
- **REPOSITORY_GUIDE.md** - Key Components section
- **ARCHITECTURE_DIAGRAM.md** - DTA Module Deep Dive

### "How do I change training parameters?"
- **QUICK_START.md** - Key Parameters table
- **REPOSITORY_GUIDE.md** - Training Parameters Explained table

### "What is a Spiking Neural Network?"
- **QUICK_START.md** - Simple Explanations section
- **REPOSITORY_GUIDE.md** - Key Concepts section
- **ARCHITECTURE_DIAGRAM.md** - LIF Neuron Operation

### "How do I modify the code?"
- **NO_PR_GUIDE.md** - Scenario 3
- **REPOSITORY_GUIDE.md** - How to Experiment section

### "I'm getting an error, help!"
- **QUICK_START.md** - Troubleshooting section
- **REPOSITORY_GUIDE.md** - Troubleshooting section

### "What are the expected results?"
- **QUICK_START.md** - Expected Results section
- **REPOSITORY_GUIDE.md** - Performance Expectations table

---

## 💡 Key Concepts Across All Docs

### 1. No PRs Needed
**Explained in:** NO_PR_GUIDE.md, QUICK_START.md
- You can use this repository completely locally
- PRs are only for contributing back (optional)
- Download, use, modify freely on your computer

### 2. Spiking Neural Networks (SNNs)
**Explained in:** All guides
- Brain-inspired neural networks
- Use discrete spikes over time instead of continuous values
- More energy-efficient than traditional neural networks

### 3. DTA (Dual Temporal-channel-wise Attention)
**Explained in:** All guides
- Innovation of this repository
- Combines two attention mechanisms: T_NA and T_XA
- Helps SNNs focus on important information
- Improves accuracy while maintaining SNN benefits

### 4. Time Steps
**Explained in:** All guides
- SNNs process data over multiple time steps (T=6 typically)
- Each image is repeated and processed T times
- Neurons accumulate information over time
- Final output is average of all time steps

### 5. Training Process
**Explained in:** REPOSITORY_GUIDE.md, ARCHITECTURE_DIAGRAM.md
- Standard supervised learning
- Uses data augmentation (Mixup, CutMix)
- CrossEntropyLoss for classification
- SGD optimizer with learning rate scheduling
- Saves best model based on validation accuracy

---

## 📊 At a Glance Comparison

| Document | Length | Technical Level | Best For |
|----------|--------|----------------|----------|
| **NO_PR_GUIDE.md** | Medium | Beginner | Understanding PRs and setup |
| **QUICK_START.md** | Short | Beginner-Intermediate | Getting started quickly |
| **REPOSITORY_GUIDE.md** | Long | Intermediate-Advanced | Deep understanding |
| **ARCHITECTURE_DIAGRAM.md** | Long | Intermediate-Advanced | Visual learners |
| **README.md** | Short | All levels | Official reference |

---

## 🎓 Learning Objectives

After reading these documents, you will understand:

✅ How to use this repository without creating pull requests
✅ What DTA-SNN is and why it matters
✅ How Spiking Neural Networks work
✅ What the DTA attention mechanism does
✅ How to train models on different datasets
✅ How to modify and experiment with the code
✅ The complete architecture from input to output
✅ How to troubleshoot common issues
✅ Expected performance and training times

---

## 🚀 Next Steps

1. **Choose your reading path** (see above)
2. **Read the relevant documents**
3. **Download/clone the repository**
4. **Install dependencies**
5. **Run your first training**
6. **Experiment and explore!**

---

## 📬 Additional Resources

- **Paper:** [WACV 2025 Conference Paper](https://openaccess.thecvf.com/content/WACV2025/papers/Kim_DTA_Dual_Temporal-Channel-Wise_Attention_for_Spiking_Neural_Networks_WACV_2025_paper.pdf)
- **SpikingJelly Library:** [GitHub](https://github.com/fangwei123456/spikingjelly)
- **PyTorch Documentation:** [pytorch.org](https://pytorch.org/docs/)

---

## 📝 Citation

If you use this code in your research, please cite:

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

## 🎉 You're Ready!

You now have everything you need to:
- Understand the repository
- Use it without creating PRs
- Train your own models
- Experiment with the code
- Learn about Spiking Neural Networks

**Happy learning and experimenting! 🚀**

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│         DTA-SNN QUICK REFERENCE                         │
├─────────────────────────────────────────────────────────┤
│ SETUP:                                                  │
│   git clone https://github.com/biswadeep20666/DTA-SNN  │
│   cd DTA-SNN                                           │
│   pip install -r requirements.txt                      │
├─────────────────────────────────────────────────────────┤
│ TRAIN:                                                  │
│   bash ./scripts/run_cifar10.sh                        │
├─────────────────────────────────────────────────────────┤
│ DOCS:                                                   │
│   NO_PR_GUIDE.md    → How to use without PRs          │
│   QUICK_START.md    → Fast reference                   │
│   REPOSITORY_GUIDE  → Complete guide                   │
│   ARCHITECTURE_DIAGRAM → Visual diagrams               │
├─────────────────────────────────────────────────────────┤
│ KEY CONCEPTS:                                           │
│   SNN = Brain-like neural networks with spikes         │
│   DTA = Attention mechanism for SNNs                   │
│   T=6 = Number of time steps for processing           │
│   No PRs = Use locally, no need to contribute back    │
└─────────────────────────────────────────────────────────┘
```

---

*Last Updated: 2025*
*Documentation created to help users understand and use DTA-SNN without creating pull requests*
