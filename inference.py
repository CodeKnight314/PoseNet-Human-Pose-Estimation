from model import ResNetPose
import argparse
import torch 
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob 
from tqdm import tqdm
import os
from torchvision import transforms as T
import numpy as np

def rescale_keypoints(keypoints, orig_size, new_size):
    orig_width, orig_height = orig_size
    new_width, new_height = new_size
    scale_x = orig_width / new_width
    scale_y = orig_height / new_height
    
    rescaled_keypoints = keypoints * np.array([scale_x, scale_y])
    return rescaled_keypoints

def inference(input_dir: str, output_dir: str, path: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNetPose().to(device)
    if path:
        model.load_state_dict(torch.load(path, map_location=device))
    
    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor()
    ])

    img_dir = glob(os.path.join(input_dir, "*.jpg"))
    for img_path in tqdm(img_dir):
        img = Image.open(img_path).convert("RGB")
        orig_size = img.size  # (width, height)
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(img_tensor).squeeze(0).cpu().detach().numpy()  # Shape: (17, 2)
        
        # Rescale keypoints to original image size
        keypoints = rescale_keypoints(prediction, orig_size, (512, 512))
        
        # Plot keypoints on original image
        plt.imshow(img)
        for x, y in keypoints:
            plt.plot(x, y, 'ro')
        plt.savefig(os.path.join(output_dir, os.path.basename(img_path)))

    print("Completed inference")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--weight")
    args = parser.parse_args()

    inference(args.input_dir, args.output_dir, args.weight)