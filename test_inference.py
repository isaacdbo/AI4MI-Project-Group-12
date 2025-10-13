#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ENet import ENet
from utils import probs2class, save_images
from post_processing import postprocess_per_class

class SimpleTestDataset(Dataset):
    def __init__(self, img_dir):
        self.img_paths = sorted(Path(img_dir).glob("*.png"))
        print(f"Found {len(self.img_paths)} images in {img_dir}")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("L")
        img_array = np.array(img)[np.newaxis, ...] / 255.0
        img_tensor = torch.tensor(img_array, dtype=torch.float32)
        
        return {
            "images": img_tensor,
            "stems": img_path.stem
        }

def main():
    parser = argparse.ArgumentParser(description="Test inference pipeline on validation set")
    parser.add_argument("--weights", type=Path, required=True, 
                        help="Path to model weights (.pt or .pkl)")
    parser.add_argument("--img_dir", type=Path, required=True,
                        help="e.g., data/SEGTHOR/val/img")
    parser.add_argument("--dest", type=Path, required=True,
                        help="Output directory for predictions")
    parser.add_argument("--arch", default="enet", 
                        choices=["enet", "enet_se", "segformer_b0"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Weights: {args.weights}")
    print(f"Images: {args.img_dir}")
    print(f"Output: {args.dest}")

    K = 5
    
    if args.arch == "enet":
        net = ENet(1, K, kernels=8, factor=2, use_se=False)
    elif args.arch == "enet_se":
        net = ENet(1, K, kernels=8, factor=2, use_se=True)
    elif args.arch == "segformer_b0":
        from segformer_b0 import SegFormerB0
        net = SegFormerB0(num_classes=K, in_ch=1)
    
    if args.weights.suffix == ".pt":
        net.load_state_dict(torch.load(args.weights, map_location=device))
    elif args.weights.suffix == ".pkl":
        net = torch.load(args.weights, map_location=device)
    else:
        raise ValueError(f"Unsupported weight format: {args.weights.suffix}")
    
    net.to(device)
    net.eval()
    
    test_set = SimpleTestDataset(args.img_dir)
    test_loader = DataLoader(
        test_set, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        shuffle=False
    )
    
    pred_dir = args.dest
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running inference on {len(test_set)} slices...")
    print(f"Batch size: {args.batch_size}\n")
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img = data["images"].to(device)
            stems = data["stems"]
            
            pred_logits = net(img)
            pred_probs = F.softmax(pred_logits, dim=1)
            predicted_class = probs2class(pred_probs)
            
            for b in range(predicted_class.shape[0]):
                pred_np = predicted_class[b, 0].cpu().numpy()
                pred_np = postprocess_per_class(pred_np)
                predicted_class[b, 0] = torch.from_numpy(pred_np)
            
            save_images(predicted_class * 63, stems, pred_dir)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {(i+1)*args.batch_size}/{len(test_set)} slices")
    

    print(f"✓ PNG predictions saved to: {pred_dir}")

if __name__ == "__main__":
    main()