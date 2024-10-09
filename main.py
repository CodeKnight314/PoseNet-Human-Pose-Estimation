from model import get_ResNetPose
from dataset import get_HPESingle
from loss import SmoothL1Loss
from utils.early_stop import EarlyStopMechanism
from utils.log_writer import LOGWRITER

import argparse
import os
import torch
import torch.optim as optim
from tqdm import tqdm

def load_optimizer(model: torch.nn.Module, opt: str, lr: float):
    if opt == "Adam":
        return optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
    elif opt == "SGD":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    elif opt == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, momentum=0.9, weight_decay=1e-5)
    elif opt == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
    elif opt == "Adadelta":
        return optim.Adadelta(model.parameters(), lr=lr, rho=0.9, weight_decay=1e-5)
    elif opt == "Adagrad":
        return optim.Adagrad(model.parameters(), lr=lr, weight_decay=1e-5)
    elif opt == "RAdam":
        return optim.RAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt}")

def get_predicted_coords(heatmaps):
    """
    """
    batch_size, num_keypoints, height, width = heatmaps.shape
    
    x_grid = torch.linspace(0, width - 1, steps=width).view(1, 1, 1, width).to(heatmaps.device)
    y_grid = torch.linspace(0, height - 1, steps=height).view(1, 1, height, 1).to(heatmaps.device)

    x = (heatmaps * x_grid).sum(dim=(2, 3))
    y = (heatmaps * y_grid).sum(dim=(2, 3))

    coordinates = torch.stack((x, y), dim=2)
    
    return coordinates

def HPE(model, optimizer, lr_scheduler, train_dl, val_dl, epochs, device, output_dir):
    criterion = SmoothL1Loss(alpha=0.5, beta=0.5)
    
    logger = LOGWRITER(output_directory=output_dir, total_epochs=epochs)
    es_mech = EarlyStopMechanism(metric_threshold=0.05, save_path=os.path.join(output_dir, "saved_weights"))
    
    try:
        logger.write(f"Starting training on device {device} for {epochs} epochs.")
        for epoch in range(epochs):
            logger.write(f"Epoch {epoch+1}/{epochs} started.")
            model.train() 
            total_train_loss = 0.0
            for i, data in enumerate(tqdm(train_dl, desc=f"[Training HourGlass] [{epoch+1}/{epochs}]: ")):
                img, ann = data 
                img, ann = img.to(device), ann.to(device)
                
                optimizer.zero_grad()
                prediction = model(img)
                prediction = get_predicted_coords(prediction)
                
                loss = criterion(prediction, ann)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            model.eval()
            total_valid_loss = 0.0
            with torch.no_grad(): 
                for i, data in enumerate(tqdm(val_dl, desc=f"[Validating HourGlass] [{epoch+1}/{epochs}]: ")):
                    img, ann = data 
                    img, ann = img.to(device), ann.to(device)
                    
                    prediction = model(img)
                    prediction = get_predicted_coords(prediction)
                    
                    loss = criterion(prediction, ann)
                    total_valid_loss += loss.item()
            
            avg_tr_loss = total_train_loss / len(train_dl)
            avg_val_loss = total_valid_loss / len(val_dl)
            
            # Early stopping mechanism
            es_mech.step(model=model, metric=avg_val_loss)
            if es_mech.check():
                logger.write("[INFO] Early Stopping Mechanism Engaged. Training procedure ended early.")
                break
            
            # Logging results
            logger.log_results(epoch=epoch+1, tr_loss=avg_tr_loss, val_loss=avg_val_loss)
            
            # Stepping scheduler
            lr_scheduler.step()
        logger.write("Training completed.")
    except Exception as e:
        logger.log_error(f"An error occurred during training: {str(e)}")
        raise e

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="root directory that hosts both train and val datasets with annotations")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory for results and weights")
    parser.add_argument("--patch_size", type=int, default=256, help="patch size for received training samples")
    parser.add_argument("--model_pth", type=str, help="Model weights for model if available")
    parser.add_argument("--opt", type=str, default="Adam", help="Optimizer for model")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for model")
    parser.add_argument("--epochs", type=int, required=True, help="Total epochs for running model")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = get_ResNetPose(args.model_pth)
    
    optimizer = load_optimizer(model, args.opt, args.lr)
    
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=args.lr/100)
    
    train_dl = get_HPESingle(root_dir=args.root_dir, mode="train", patch_size=args.patch_size, batch_size=16)
    val_dl = get_HPESingle(root_dir=args.root_dir, mode="val", patch_size=args.patch_size, batch_size=16)
    
    HPE(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, train_dl=train_dl, val_dl=val_dl, epochs=args.epochs, device=device, output_dir=args.output_dir)