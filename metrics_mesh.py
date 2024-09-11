from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import LPIPS  # Correct import
import json
from tqdm import tqdm
from utils.image_utils import psnr

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []

    render_files = sorted([f for f in os.listdir(renders_dir) if f.endswith(".png")])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".png")])

    if len(render_files) != len(gt_files):
        raise ValueError("The number of images in the render directory and ground truth directory must be the same.")

    for render_file, gt_file in zip(render_files, gt_files):
        render_path = renders_dir / render_file
        gt_path = gt_dir / gt_file

        render = Image.open(render_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGBA")

        # Get the dimensions of both images
        render_width, render_height = render.size
        gt_width, gt_height = gt.size

        # Find the minimum width and height
        min_width = min(render_width, gt_width)
        min_height = min(render_height, gt_height)

        # Resize both images to the minimum dimensions
        render = render.resize((min_width, min_height), Image.Resampling.LANCZOS)
        gt = gt.resize((min_width, min_height), Image.Resampling.LANCZOS)

        render_tensor = tf.to_tensor(render).cuda()
        gt_tensor = tf.to_tensor(gt).cuda()

        # Extract alpha channel and use it as a mask
        alpha_mask = gt_tensor[3:4, :, :]  # Extract alpha channel
        mask = alpha_mask > 0  # Create binary mask where alpha > 0

        # Apply mask to ignore the background in GT and test images
        gt_tensor = gt_tensor[:3, :, :] * mask
        render_tensor = render_tensor * mask

        renders.append(render_tensor.unsqueeze(0))
        gts.append(gt_tensor.unsqueeze(0))
    
    return renders, gts

def evaluate(gt_dir, test_dir):
    gt_dir = Path(gt_dir)
    test_dir = Path(test_dir)

    full_dict = {}
    per_view_dict = {}

    print("Evaluating images...")
    
    renders, gts = readImages(test_dir, gt_dir)

    ssims = []
    psnrs = []
    lpipss = []

    lpips_model = LPIPS(net_type='vgg').cuda() 

    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
        ssims.append(ssim(renders[idx], gts[idx]).item())  
        psnrs.append(psnr(renders[idx], gts[idx]).item()) 
        lpipss.append(lpips_model(renders[idx], gts[idx]).item())  

    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean()))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean()))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean()))
    print("")

    full_dict.update({"SSIM": torch.tensor(ssims).mean().item(),
                      "PSNR": torch.tensor(psnrs).mean().item(),
                      "LPIPS": torch.tensor(lpipss).mean().item()})

    # No need for .item() as these are already floats
    per_view_dict.update({"SSIM": ssims,
                          "PSNR": psnrs,
                          "LPIPS": lpipss})

    # Save results
    results_file = test_dir.parent / "results.json"
    per_view_file = test_dir.parent / "per_view.json"

    with open(results_file, 'w') as fp:
        json.dump(full_dict, fp, indent=True)
    with open(per_view_file, 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)

    print(f"Results saved to {results_file} and {per_view_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate images with SSIM, PSNR, and LPIPS ignoring background.")
    parser.add_argument('--gt_dir', type=str, required=True, help="Directory containing ground truth images (with alpha transparency).")
    parser.add_argument('--test_dir', type=str, required=True, help="Directory containing test images (with black background).")
    
    args = parser.parse_args()
    evaluate(args.gt_dir, args.test_dir)
