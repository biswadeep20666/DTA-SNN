# SpikingJelly Backend Analysis

## Overview

This document provides a comprehensive analysis of the SpikingJelly framework and confirms that **SpikingJelly uses PyTorch as its computational backend**.

## What is SpikingJelly?

SpikingJelly is a deep learning framework for Spiking Neural Networks (SNNs) built on top of PyTorch. It provides:
- Ready-to-use spiking neuron models (LIF, IF, etc.)
- Neuromorphic dataset loaders
- Surrogate gradient functions for backpropagation
- Utilities for temporal sequence processing

## Backend Confirmation

### 1. PyTorch as the Primary Backend

SpikingJelly is **entirely built on PyTorch**. All core components inherit from PyTorch classes:

- **Neuron models**: Inherit from `torch.nn.Module`
  - Example: `LIFNode`, `IFNode`, `ParametricLIFNode`
  - All neuron dynamics are implemented using PyTorch operations
  
- **Datasets**: Inherit from `torch.utils.data.Dataset`
  - Example: `CIFAR10DVS`, `DVS128Gesture`
  - Compatible with PyTorch's DataLoader

- **All operations use PyTorch tensors**: Every computation in SpikingJelly operates on `torch.Tensor` objects

### 2. Multiple Backend Support

While PyTorch is the primary backend, SpikingJelly also supports:

- **`torch`** (default): Pure PyTorch implementation
- **`cupy`**: Accelerated CUDA kernels using CuPy
- **`lava`**: Intel's Lava neuromorphic framework

The backend is configurable via the `backend` parameter in neuron initialization:

```python
from spikingjelly.activation_based import neuron

# Default PyTorch backend
lif_node = neuron.LIFNode(backend='torch')

# CuPy-accelerated backend
lif_node = neuron.LIFNode(backend='cupy')
```

### 3. Source Code Evidence

From the SpikingJelly source code (`spikingjelly/activation_based/neuron.py`):

```python
import torch
import torch.nn as nn
from . import surrogate, base
from .. import configure

class LIFNode(BaseNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, 
                 v_threshold: float = 1., v_reset: float = 0., 
                 surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', 
                 backend='torch',  # Default backend is PyTorch
                 store_v_seq: bool = False):
        # Implementation uses torch operations
```

## How This Project (DTA-SNN) Uses SpikingJelly

This project uses SpikingJelly in **two specific ways**:

### 1. Dataset Loading

```python
# From main.py
from spikingjelly.datasets import cifar10_dvs

# Used to load DVS-CIFAR10 neuromorphic dataset
origin_set = cifar10_dvs.CIFAR10DVS(
    root="./dataset/DVS_CIFAR10", 
    data_type='frame', 
    frames_number=args.time_step, 
    split_by='number'
)
```

### 2. Utility Code Adaptation

```python
# From models/layers.py
class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)
```

This utility class flattens the temporal dimension to apply ANN operations across all time steps.

## Custom SNN Implementation

**Important Note**: While this project uses SpikingJelly for datasets and utilities, it implements its **own custom SNN components**:

- **Custom LIFSpike neuron** (`models/layers.py`):
  ```python
  class LIFSpike(nn.Module):
      def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
          super(LIFSpike, self).__init__()
          self.act = ZIF.apply
          self.thresh = thresh
          self.tau = tau
          self.gama = gama
  ```

- **Custom surrogate gradient function** (ZIF):
  ```python
  class ZIF(torch.autograd.Function):
      @staticmethod
      def forward(ctx, input, gama):
          out = (input > 0).float()
          # ...
      
      @staticmethod
      def backward(ctx, grad_output):
          # Custom gradient computation
          # ...
  ```

This custom implementation is **also built directly on PyTorch**, using:
- `torch.nn.Module` for neuron models
- `torch.autograd.Function` for custom gradients
- `torch.Tensor` operations for all computations

## Conclusion

### Yes, SpikingJelly uses PyTorch as its backend.

**Key Points:**
1. SpikingJelly is a PyTorch-based framework for spiking neural networks
2. All computations use PyTorch tensors and operations
3. Neuron models inherit from `torch.nn.Module`
4. Datasets inherit from `torch.utils.data.Dataset`
5. The default and primary backend is `'torch'` (PyTorch)
6. This project uses SpikingJelly for dataset loading and utilities, but implements custom SNN components directly in PyTorch

**PyTorch Version in This Project:**
- As specified in `Dockerfile`: PyTorch 1.13.1 with CUDA 11.6
- Fully compatible with SpikingJelly's PyTorch backend

## References

- SpikingJelly GitHub: https://github.com/fangwei123456/spikingjelly
- SpikingJelly Documentation: https://spikingjelly.readthedocs.io/
- PyTorch: https://pytorch.org/

---

*Analysis performed on 2025-10-15*
