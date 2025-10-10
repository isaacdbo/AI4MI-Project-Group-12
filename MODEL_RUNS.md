# Model Training Guide

## Model Results Directory Structure

All model training results are saved in the `model_results/` directory, organized by dataset and architecture:

```
model_results/
└── segthor_clean/
    ├── enet/
    ├── enet_se/
    └── segformer_b0/
```

Each model run directory contains:
- `bestmodel.pkl` - Complete saved model (architecture + weights)
- `bestweights.pt` - Model weights only
- `best_epoch.txt` - Information about the best performing epoch
- `best_epoch/` - Predictions from the best epoch
- `loss_tra.npy` - Training loss history
- `loss_val.npy` - Validation loss history
- `dice_tra.npy` - Training Dice coefficient history
- `dice_val.npy` - Validation Dice coefficient history
- `iter{XXX}/` - Predictions from each epoch

## Available Model Architectures

The following architectures are available via the `--arch` flag:

1. **enet** - Standard ENet architecture
2. **enet_se** - ENet with Squeeze-and-Excitation blocks
3. **segformer_b0** - SegFormer-B0 (MiT encoder + lightweight decoder)

## Training Commands

### Basic Training Command Structure

```bash
python main.py --dataset SEGTHOR_CLEAN --mode full --dest <output_directory> --arch <architecture> --epochs <num_epochs> [--gpu]
```

### Specific Model Training Commands

#### 1. ENet (Standard)
```bash
python main.py --dataset SEGTHOR_CLEAN --mode full --dest model_results/segthor_clean/enet/base --arch enet --epochs 20 --gpu
```

#### 2. ENet with Squeeze-and-Excitation
```bash
python main.py --dataset SEGTHOR_CLEAN --mode full --dest model_results/segthor_clean/enet_se/base --arch enet_se --epochs 20 --gpu
```

#### 3. SegFormer-B0
```bash
python main.py --dataset SEGTHOR_CLEAN --mode full --dest model_results/segthor_clean/segformer_b0/base --arch segformer_b0 --epochs 20 --gpu
```

### Training Options

- `--dataset`: Dataset to use (`SEGTHOR_CLEAN`, `SEGTHOR`, `TOY2`)
- `--mode`: Training mode (`full` or `partial`)
  - `full`: Supervise all classes including background
  - `partial`: For SEGTHOR dataset, excludes heart class (class 2) from supervision
- `--dest`: Output directory for results (required)
- `--arch`: Model architecture (default: `enet`)
- `--epochs`: Number of training epochs (default: 20)
- `--gpu`: Use GPU if available (recommended)
- `--debug`: Use only 10 samples for quick testing

## Requirements

Ensure you have:
- PyTorch installed
- Data located in `data/SEGTHOR_CLEAN/` directory
- Sufficient GPU memory if using `--gpu` flag
