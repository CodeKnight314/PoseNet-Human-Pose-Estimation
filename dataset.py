import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import torchvision.transforms as T

from PIL import Image
import numpy as np
import os
import json
from collections import defaultdict
from typing import Dict, List
from glob import glob

# For unaltered COCO dataset with multiple HPE targets and BBox
class HPEDataset(Dataset):
    def __init__(self, root_dir : str, mode : str, patch_size : int):
        super().__init__()
        self.patch_size = patch_size
        
        annFile = os.path.join(root_dir, "annotations", f"person_keypoints_{mode}2017.json")
        with open(annFile, 'r') as file: 
            self.annotations = json.load(file)
        
        self.image_id_to_info = {img['id']: img for img in self.annotations['images']}
        
        self.annotations_per_image = defaultdict(list)
        for ann in self.annotations['annotations']:
            self.annotations_per_image[ann['image_id']].append(ann)
        
        self.img_ids = list(self.image_id_to_info.keys())
        
        self.img_dir = os.path.join(root_dir, f"{mode}2017")
        
        self.img_transform = T.Compose([
            T.ColorJitter(),
            T.GaussianBlur(kernel_size=(3, 3)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]) 

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index : int) -> List[torch.Tensor, Dict[str : torch.Tensor]]:
        img_id = self.img_ids[index]
        img_info = self.image_id_to_info[img_id]
        file_name = img_info['file_name']
        width, height = img_info['width'], img_info['height']
        
        img_path = os.path.join(self.img_dir, file_name)
        img = Image.open(img_path).convert("RGB")
        
        img = img.resize((self.patch_size, self.patch_size), Image.BICUBIC)
        
        scale_x = self.patch_size / width
        scale_y = self.patch_size / height
        
        annotations = self.annotations_per_image[img_id]
        
        keypoints_list = []
        boxes_list = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in annotations:
            if ann['num_keypoints'] > 0:
                keypoints = np.array(ann['keypoints']).reshape(-1, 2)
                
                keypoints[:, 0] *= scale_x
                keypoints[:, 1] *= scale_y
                keypoints_list.append(keypoints)
                
                bbox = ann['bbox']
                x_min, y_min, box_width, box_height = bbox
                x_max = x_min + box_width
                y_max = y_min + box_height
                
                x_min *= scale_x
                y_min *= scale_y
                x_max *= scale_x
                y_max *= scale_y
                boxes_list.append([x_min, y_min, x_max, y_max])
                
                labels.append(ann['category_id'])
                areas.append(ann['area'] * scale_x * scale_y)
                iscrowd.append(ann['iscrowd'])
        
        if len(keypoints_list) == 0:
            keypoints = torch.zeros((0, 17, 3), dtype=torch.float32)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            keypoints = torch.tensor(keypoints_list, dtype=torch.float32)  
            boxes = torch.tensor(boxes_list, dtype=torch.float32)          
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        
        img = self.img_transform(img)
        
        target = {
            'keypoints': keypoints,
            'boxes': boxes,
            'labels': labels,
            'areas': areas,
            'iscrowd': iscrowd,
            'image_id': torch.tensor([img_id])
        }
        
        return img, target

class HPESingle(Dataset): 
    def __init__(self, root_dir : str, mode : str, patch_size : int): 
        super().__init__() 
        
        self.img_dir = glob(os.path.join(root_dir, f"{mode}/*.jpg"))
        self.ann_dir = glob(os.path.join(root_dir, f"{mode}/*.txt"))
        
        self.transform = T.Compose([
            T.Resize((patch_size, patch_size)), 
            T.GaussianBlur(kernel_size=(3,3)),
            T.ToTensor()
        ])
        
        self.patch_size = patch_size
            
    def __len__(self): 
        return len(self.img_dir)
    
    def __getitem__(self, index):
        img = Image.open(self.img_dir[index]).convert("RGB")
        width, height = img.size
        
        scale_width, scale_height = self.patch_size / width, self.patch_size / height

        with open(self.ann_dir[index], 'r') as f:
            data = [[float(x) for x in line.strip().split()] for line in f]
            data[:, 0] *= scale_width
            data[:, 1] *- scale_height
        
        img_tensor = self.transform(img)
        ann_tensor = torch.Tensor(data)
        
        return img_tensor, ann_tensor  
    
def get_HPESingle(root_dir : str, mode : str, patch_size : int, batch_size : int):
    """
    Helper Function for getting HPESingle Dataset
    
    Args: 
        root_dir (str): directory containing train/ and val/ folders. 
        mode (str): to specify which train/val/test split to use from root_dir
        patch_size (int): dimensions to resize HPE image into
        batch_size (int): number of training samples per iteration
        
    Return: 
        DataLoader: DataLoader for training/validating model
    """
    dataset = HPEDataset(root_dir, mode, patch_size)
    return DataLoader(dataset, batch_size, shuffle=True, num_workers=os.cpu_count())