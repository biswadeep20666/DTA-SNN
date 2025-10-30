"""Quick test to verify dataset and model setup."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from datasets.etram_dataset import build_etram_dataloader
from models.dta_snn_seq2seq import build_dta_snn_seq2seq

def test_dataset():
    """Test dataset loading."""
    print("=" * 80)
    print("Testing ETram Dataset")
    print("=" * 80)
    
    data_root = "/home/biswadeep/OpenSTL/data/etram_npz_binary_30hz_crop512_runs"
    
    try:
        # Build dataloaders
        train_loader = build_etram_dataloader(
            data_root, "train", pre_seq=10, aft_seq=10,
            batch_size=2, stride=1, num_workers=0
        )
        
        # Get one batch
        inputs, targets = next(iter(train_loader))
        
        print(f"✓ Train loader created successfully")
        print(f"  Input shape: {inputs.shape}")  # Should be [B, 10, 2, H, W]
        print(f"  Target shape: {targets.shape}")  # Should be [B, 10, 2, H, W]
        print(f"  Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
        print(f"  Target range: [{targets.min():.3f}, {targets.max():.3f}]")
        
        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False


def test_model():
    """Test model forward pass."""
    print("\n" + "=" * 80)
    print("Testing DTA-SNN Model")
    print("=" * 80)
    
    try:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Build model
        model = build_dta_snn_seq2seq(
            pre_seq=10,
            aft_seq=10,
            in_channels=2,
            out_channels=2,
            encoder_time=10,
            DTA_ON=True,
        ).to(device)
        
        print(f"✓ Model created successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        
        # Test forward pass
        dummy_input = torch.rand(2, 10, 2, 128, 128).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")  # Should be [2, 10, 2, 128, 128]
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """Test one training step."""
    print("\n" + "=" * 80)
    print("Testing Training Step")
    print("=" * 80)
    
    try:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        
        # Build model and optimizer
        model = build_dta_snn_seq2seq(
            pre_seq=10, aft_seq=10, in_channels=2, out_channels=2,
            encoder_time=10, DTA_ON=True,
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Load one batch
        data_root = "/home/biswadeep/OpenSTL/data/etram_npz_binary_30hz_crop512_runs"
        train_loader = build_etram_dataloader(
            data_root, "train", pre_seq=10, aft_seq=10,
            batch_size=2, stride=1, num_workers=0
        )
        
        inputs, targets = next(iter(train_loader))
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training step successful")
        print(f"  Loss: {loss.item():.6f}")
        
        return True
    except Exception as e:
        print(f"✗ Training step test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\nRunning DTA-SNN Setup Tests\n")
    
    results = {
        "Dataset": test_dataset(),
        "Model": test_model(),
        "Training": test_training_step(),
    }
    
    print("\n" + "=" * 80)
    print("Test Results Summary")
    print("=" * 80)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:20s}: {status}")
    
    all_passed = all(results.values())
    print("=" * 80)
    
    if all_passed:
        print("\n✓ All tests passed! Ready to train.")
        print("\nTo start training, run:")
        print("  bash scripts/train_etram.sh")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
    
    sys.exit(0 if all_passed else 1)
