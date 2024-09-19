from PIL import Image 
import json 
import argparse
import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm

def main(root_dir: str, mode: str, annotations_dir: str):
    patch_size = 256
    
    with open(annotations_dir, 'r') as file:
        annotations = json.load(file)

    img_dir = os.path.join(root_dir, f"{mode}2017")

    images_dir = os.path.join(root_dir, f'{mode}_cropped_images')
    keypoints_dir = os.path.join(root_dir, f'{mode}_keypoints')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(keypoints_dir, exist_ok=True)

    image_id_to_info = {img['id']: img for img in annotations['images']}

    annotations_per_image = defaultdict(list)
    for ann in annotations['annotations']:
        annotations_per_image[ann['image_id']].append(ann)

    iteration = 0

    for img_id in tqdm(image_id_to_info.keys()):
        img_info = image_id_to_info[img_id]
        file_name = img_info['file_name']
        width, height = img_info['width'], img_info['height']
        img_path = os.path.join(img_dir, file_name)
        img = Image.open(img_path).convert("RGB")

        annotations_for_img = annotations_per_image[img_id]

        for ann in annotations_for_img:
            if ann['num_keypoints'] > 15:
                keypoints = np.array(ann['keypoints']).reshape(-1, 3)

                bbox = ann['bbox']
                x_min, y_min, box_width, box_height = bbox
                x_max = x_min + box_width
                y_max = y_min + box_height
                
                scale_width, scale_height = patch_size / box_width, patch_size / box_height

                img_crop = img.crop((x_min, y_min, x_max, y_max)).resize((patch_size, patch_size))

                img_filename = f"{str(iteration).zfill(7)}.jpg"
                img_crop.save(os.path.join(images_dir, img_filename))

                kp_filename = f"{str(iteration).zfill(7)}.txt"
                with open(os.path.join(keypoints_dir, kp_filename), 'w') as kp_file:
                    for kp in keypoints:
                        kp_file.write(f"{kp[0] * scale_width} {kp[1] * scale_height} {kp[2]}\n")

                iteration += 1

    print(f"Total cropped images: {iteration}")
    
if __name__ == "__main__": 
    parser =argparse.ArgumentParser() 
    parser.add_argument("--root_dir", type=str, required=True, help="Directory to the root directory")
    parser.add_argument("--mode", type=str, required=True, help="mode for COCO dataset")
    parser.add_argument("--ann_dir", type=str, required=True, help="Directory to annotations json")
    
    args = parser.parse_args()
    
    main(args.root_dir, args.mode, args.ann_dir)