#!/bin/bash
# Complete setup verification and first run guide for DTA-SNN ETram training

echo "════════════════════════════════════════════════════════════════════════════════"
echo "DTA-SNN ETram Training Setup Verification"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check dataset
echo "Checking dataset..."
DATA_ROOT="/home/biswadeep/OpenSTL/data/etram_npz_binary_30hz_crop512_runs"

if [ ! -d "$DATA_ROOT" ]; then
    echo -e "${RED}✗ Dataset directory not found: $DATA_ROOT${NC}"
    exit 1
fi

for split in train val test; do
    count=$(ls $DATA_ROOT/$split/*.npz 2>/dev/null | wc -l)
    if [ $count -gt 0 ]; then
        echo -e "${GREEN}✓ $split split: $count files${NC}"
    else
        echo -e "${RED}✗ $split split: no files found${NC}"
    fi
done

echo ""

# Check Python dependencies
echo "Checking Python environment..."
python -c "
import sys
missing = []
try:
    import torch
    print(f'✓ PyTorch: {torch.__version__}')
except:
    missing.append('torch')
    print('✗ PyTorch not found')

try:
    import numpy
    print(f'✓ NumPy: {numpy.__version__}')
except:
    missing.append('numpy')
    print('✗ NumPy not found')

try:
    import tqdm
    print(f'✓ tqdm installed')
except:
    missing.append('tqdm')
    print('✗ tqdm not found')

try:
    import matplotlib
    print(f'✓ Matplotlib installed')
except:
    missing.append('matplotlib')
    print('✗ Matplotlib not found (optional, for visualization)')

if missing and 'matplotlib' not in missing:
    print(f'\nMissing required packages: {missing}')
    print('Install with: pip install ' + ' '.join(missing))
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Python dependencies check failed${NC}"
    exit 1
fi

echo ""

# Check GPU
echo "Checking GPU availability..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'  GPU 1: {torch.cuda.get_device_name(1) if torch.cuda.device_count() > 1 else \"Not available\"}')
else:
    print('⚠ CUDA not available (CPU training will be very slow)')
"

echo ""

# Check DTA-SNN files
echo "Checking DTA-SNN files..."
files=(
    "datasets/__init__.py"
    "datasets/etram_dataset.py"
    "models/dta_snn_seq2seq.py"
    "train_etram.py"
    "evaluate_etram.py"
    "test_setup.py"
    "scripts/train_etram.sh"
    "scripts/eval_etram.sh"
    "README_ETRAM.md"
)

all_found=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ $file${NC}"
    else
        echo -e "${RED}✗ $file (missing)${NC}"
        all_found=false
    fi
done

echo ""

if ! $all_found; then
    echo -e "${RED}✗ Some files are missing. Please re-run the setup.${NC}"
    exit 1
fi

echo "════════════════════════════════════════════════════════════════════════════════"
echo -e "${GREEN}✓ Setup verification completed successfully!${NC}"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Ask user if they want to run tests
echo "Would you like to run the setup test now? (y/n)"
read -r response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo ""
    echo "Running setup tests..."
    echo ""
    python test_setup.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "════════════════════════════════════════════════════════════════════════════════"
        echo -e "${GREEN}✓ All tests passed! Ready to train.${NC}"
        echo "════════════════════════════════════════════════════════════════════════════════"
        echo ""
        echo "To start training, run ONE of these commands:"
        echo ""
        echo "  1. Standard training (with background):"
        echo -e "     ${YELLOW}bash scripts/train_etram.sh${NC}"
        echo ""
        echo "  2. Training with object masking (cleaner signal):"
        echo -e "     ${YELLOW}bash scripts/train_etram_objmask.sh${NC}"
        echo ""
        echo "  3. Background training with logging:"
        echo -e "     ${YELLOW}CUDA_VISIBLE_DEVICES=1 nohup bash scripts/train_etram.sh > logs/train.log 2>&1 &${NC}"
        echo -e "     ${YELLOW}tail -f logs/train.log${NC}"
        echo ""
        echo "See QUICKSTART.md for more options."
    else
        echo ""
        echo -e "${RED}✗ Tests failed. Please check the errors above.${NC}"
        exit 1
    fi
else
    echo ""
    echo "Skipping tests. You can run them later with:"
    echo "  python test_setup.py"
    echo ""
    echo "When ready to train, run:"
    echo "  bash scripts/train_etram.sh"
fi

echo ""
