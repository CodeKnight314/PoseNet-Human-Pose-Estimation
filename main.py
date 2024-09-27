from model import StackedHourGlass
from dataset import get_HPESingle
from loss import SmoothL1Loss
from utils.early_stop import EarlyStopMechanism
from utils.log_writer import LOGWRITER

import argparse
import os
import torch
import torch.optim as optim
from tqdm import tqdm

def load_optimizer(model : torch.nn.Module, opt : str, lr : float): 
    if(opt == "Adam"):
        return optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
    else: 
        return optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)

def HPE(model, optimizer, lr_scheduler, train_dl, val_dl, epochs, device, output_dir):
    criterion = SmoothL1Loss(alpha=0.5, beta=0.5)
    
    logger = LOGWRITER(output_directory=output_dir)
    es_mech = EarlyStopMechanism(save_path=os.path.join(output_dir, "saved_weights"))
    
    for epoch in range(epochs):
        
        model.train() 
        total_train_loss = 0.0
        for i, data in enumerate(tqdm(train_dl, desc=f"[Training HourGlass] [{epoch+1}/{epochs}]: ")):
            img, ann = data 
            img, ann = img.to(device), ann.to(device)
            
            optimizer.zero_grad()
            prediction = model(img)
            
            loss = criterion(prediction, ann)
            loss.backward()
            optimizer.step()
            
            total_train_loss+=loss.item()
        
        model.eval()
        total_valid_loss = 0.0
        with torch.no_grad(): 
            for i, data in enumerate(tqdm(val_dl, desc=f"[Validating HourGlass] [{epoch+1}/{epochs}]: ")):
                img, ann = data 
                img, ann = img.to(device), ann.to(device)
                
                prediction = model(img)
                
                loss = criterion(prediction, ann)
                total_valid_loss+=loss.item()
        
        avg_tr_loss = total_train_loss / len(train_dl)
        avg_val_loss = total_valid_loss / len(val_dl)
        
        # Early stopping mechanism
        es_mech.step(model=model, metric=avg_val_loss)
        if es_mech.check():
            logger.write("[INFO] Early Stopping Mechanism Engaged. Training procedure ended early.")
            break
        
        # Logging results
        logger.log_results(epoch=epoch+1, tr_loss=avg_tr_loss, val_loss=avg_val_loss)
        
        # Stepping scheduelr
        lr_scheduler.step()

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="root directory that hosts both train and val datasets with annotations")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory for results and weights")
    parser.add_argument("--patch_size", type=int, default=256, help="patch size for received training samples")
    parser.add_argument("--model_pth", type=str, help="Model weights for model if available")
    parser.add_argument("--opt", type=str, default="Adam", help="Optimizer for model")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for model")
    parser.add_argument("--scheduler_json", type=str, help="Json path for scheduler config and name")
    parser.add_argument("--epochs", type=int, required=True, help="Total epochs for running model")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = StackedHourGlass(input_channels=3, num_modules=4, num_blocks=2, num_depth=3, num_keypoints=17).to(device)
    
    optimizer = load_optimizer(model, args.opt, args.lr)
    
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=args.lr/100)
    
    train_dl = get_HPESingle(root_dir=args.root_dir, mode="train", patch_size=args.patch_size, batch_size=16)
    val_dl = get_HPESingle(root_dir=args.root_dir, model="val", patch_size=args.patch_size, batch_size=16)
    
    
    