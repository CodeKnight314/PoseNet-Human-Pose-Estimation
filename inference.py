from model import StackedHourGlass
import argparse
import torch 
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob 
from tqdm import tqdm
import os
from torchvision import transforms as T

def extract_keypoints_subpixel(heatmaps):
    num_keypoints, height, width = heatmaps.shape
    keypoints = []
    for j in range(num_keypoints):
        heatmap = heatmaps[j, :, :]

        max_val, max_idx = torch.max(heatmap.view(-1), 0)
        max_idx = max_idx.item()
        y, x = divmod(max_idx, width)

        x_min = max(x - 1, 0)
        x_max = min(x + 1, width - 1)
        y_min = max(y - 1, 0)
        y_max = min(y + 1, height - 1)

        window = heatmap[y_min:y_max+1, x_min:x_max+1]
        window_sum = window.sum()

        if window_sum > 0:
            dx = (window * torch.tensor([[-1, 0, 1]])).sum() / window_sum
            dy = (window * torch.tensor([[-1], [0], [1]])).sum() / window_sum
            x += dx.item()
            y += dy.item()

        keypoints.append([x, y])
        
    return keypoints

def inference(input_dir : str, output_dir : str, path : str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StackedHourGlass().to(device)
    if path: 
        model.load_state_dict(torch.load(path, map_location=device))
        
    transform = T.ToTensor()

    img_dir = glob(os.path.join(input_dir, "*.jpg"))
    for img in tqdm(img_dir):
        img_tensor = transform(Image.open(img).convert("RGB")).unsqueeze(0).to(device)
        
        with torch.no_grad(): 
            prediction = model(img_tensor).squeeze(0)
            
        keypoints = extract_keypoints_subpixel(prediction)
        
        plt.imshow(Image.open(img))
        for x, y in keypoints: 
            plt.plot(x, y, 'ro')
        plt.savefig(os.path.join(output_dir, os.path.basename(img) + ".jpg"))

    print("Completed inference")
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--weight")