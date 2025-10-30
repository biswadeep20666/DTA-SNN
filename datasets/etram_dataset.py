"""ETram NPZ dataset loader for seq-to-seq event-based prediction."""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Literal, Optional, Callable


class EtramNPZDataset(Dataset):
    """
    Loads ETram .npz run files for event-based video prediction.
    Each file contains a run of consecutive valid frames [T, 2, H, W].
    We extract sliding windows of length (pre_seq + aft_seq) with stride.
    
    Args:
        data_root: Path to dataset root (contains train/val/test subdirs)
        split: One of 'train', 'val', 'test'
        pre_seq: Number of history frames (input)
        aft_seq: Number of future frames (prediction target)
        stride: Stride for sliding window extraction
        transform: Optional transform to apply to sequences
        use_obj_mask: If True and available, apply object masks to history frames
    """
    
    def __init__(
        self,
        data_root: str,
        split: Literal["train", "val", "test"] = "train",
        pre_seq: int = 10,
        aft_seq: int = 10,
        stride: int = 1,
        transform: Optional[Callable] = None,
        use_obj_mask: bool = False,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.pre_seq = pre_seq
        self.aft_seq = aft_seq
        self.stride = stride
        self.transform = transform
        self.use_obj_mask = use_obj_mask
        self.L = pre_seq + aft_seq
        
        # Find all NPZ files in split subdirectory
        split_dir = self.data_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        self.npz_files = sorted(list(split_dir.glob("*.npz")))
        if len(self.npz_files) == 0:
            raise ValueError(f"No .npz files found in {split_dir}")
        
        # Build index: (file_idx, start_frame_in_file)
        self.samples = []
        total_frames = 0
        
        for file_idx, npz_path in enumerate(self.npz_files):
            with np.load(npz_path) as data:
                if "T" in data:
                    T = int(data["T"])
                else:
                    T = int(data["frames"].shape[0])
                
                total_frames += T
                
                # Extract windows with stride
                for start in range(0, T - self.L + 1, self.stride):
                    self.samples.append((file_idx, start))
        
        print(f"[{split}] Loaded {len(self.npz_files)} files, "
              f"{total_frames} total frames, {len(self.samples)} windows "
              f"(pre_seq={pre_seq}, aft_seq={aft_seq}, stride={stride})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_idx, start = self.samples[idx]
        
        # Load file (cached in memory if small dataset, else load per-sample)
        data = np.load(self.npz_files[file_idx])
        frames = data["frames"]  # [T, 2, H, W] uint8 {0, 1} or {0, 255}
        
        # Extract window [start : start+L]
        window = frames[start : start + self.L].astype(np.float32)
        
        # Normalize to [0, 1] if needed
        if window.max() > 1.0:
            window = window / 255.0
        
        # Split into input/target
        input_seq = window[:self.pre_seq]   # [pre_seq, 2, H, W]
        target_seq = window[self.pre_seq:]  # [aft_seq, 2, H, W]
        
        # Optionally apply object masks to history frames (no label leakage)
        if self.use_obj_mask and "obj_mask" in data:
            obj_mask = data["obj_mask"][start : start + self.pre_seq]  # [pre_seq, H, W]
            obj_mask = obj_mask.astype(np.float32)
            # Apply mask to both ON and OFF channels
            input_seq[:, 0] *= obj_mask  # ON channel
            input_seq[:, 1] *= obj_mask  # OFF channel
        
        # Convert to torch tensors
        input_seq = torch.from_numpy(input_seq)
        target_seq = torch.from_numpy(target_seq)
        
        if self.transform:
            input_seq = self.transform(input_seq)
            target_seq = self.transform(target_seq)
        
        return input_seq, target_seq


def build_etram_dataloader(
    data_root: str,
    split: str,
    pre_seq: int = 10,
    aft_seq: int = 10,
    batch_size: int = 4,
    stride: int = 1,
    num_workers: int = 4,
    shuffle: bool = None,
    use_obj_mask: bool = False,
):
    """
    Build dataloader for ETram dataset.
    
    Args:
        data_root: Path to dataset root
        split: One of 'train', 'val', 'test'
        pre_seq: Number of input frames
        aft_seq: Number of prediction frames
        batch_size: Batch size
        stride: Window extraction stride
        num_workers: Number of dataloader workers
        shuffle: Whether to shuffle (defaults to True for train, False otherwise)
        use_obj_mask: Whether to apply object masks to history frames
    
    Returns:
        DataLoader instance
    """
    if shuffle is None:
        shuffle = (split == "train")
    
    dataset = EtramNPZDataset(
        data_root=data_root,
        split=split,
        pre_seq=pre_seq,
        aft_seq=aft_seq,
        stride=stride,
        use_obj_mask=use_obj_mask,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
    )
    
    return dataloader
