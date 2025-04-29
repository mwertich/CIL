import os
import torch
import torch.nn as nn
from utils.dataloader import get_dataloaders
from utils.visualization import visualize_prediction_with_ground_truth, visualize_prediction_without_ground_truth
from utils.loss_funcs import DepthUncertaintyLoss, scale_invariant_rmse
from utils.utils import torch_seed
from model import MiDaSUQ
from evaluate import evaluate_model
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import random

# for logging
from pathlib import Path
from datetime import datetime

run_id = datetime.now().strftime("%y%m%d_%H%M%S")
print('---------------- Run id:', run_id, '----------------')

torch_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and transforms
# model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")

model = MiDaS_UQ(backbone="vitl16_384")
state_dict = torch.load("models/model_finetuned_final.pth", map_location=device)
filtered_state_dict = {
    k: v for k, v in state_dict.items()
    if "scratch.output_conv.4." not in k  # Exclude final conv layer
}
model.load_state_dict(filtered_state_dict, strict=False)
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

image_size = [426, 560]
train_loader, val_loader, test_loader = get_dataloaders(image_size=image_size)


# Main training function
def finetune_model(model, train_loader, val_loader, out_path, epochs=5, lr=1e-5):

    model.to(device)
    model.train()  # set to train mode

    # Define loss and optimizer
    # criterion = nn.L1Loss()  # or nn.MSELoss()
    criterion = DepthUncertaintyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for images, depths in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            images, depths = images.to(device), depths.to(device)

            optimizer.zero_grad()
            pred_depths, pred_logvars = model(images)

            pred_depths_resized = torch.nn.functional.interpolate(
                pred_depths.unsqueeze(1),
                size=depths.shape[-2:],
                mode="bicubic",
                align_corners=False
            ).squeeze(1)

            pred_logvars_resized = torch.nn.functional.interpolate(
                pred_logvars.unsqueeze(1),
                size=depths.shape[-2:],
                mode="bicubic",
                align_corners=False
            ).squeeze(1)

            loss = criterion(pred_depths_resized, pred_logvars_resized, depths.squeeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
        evaluate_model(model, val_loader, epoch)
        # Save model after each epoch
        
        # model_path = f"models/model_finetuned_epoch_{epoch}.pth"
        #torch.save(model.state_dict(), model_path)
        #print(f"💾 Model saved to models/{model_path}")

    # Save fine-tuned model
    torch.save(model.state_dict(), out_path)
    print(f"✅ Fine-tuned model saved to {out_path}")


# def predict_model(model, test_loader):
#     model.eval()
#     with torch.no_grad():
#         for images, out_paths in test_loader:
#             images = images.to(device)

#             preds, _ = model(images)
#             preds_resized = torch.nn.functional.interpolate(
#                 preds.unsqueeze(1), size=image_size, mode="bicubic", align_corners=False
#             )
#             eps = 1e-8
#             preds_resized = preds_resized.clamp(min=eps) # for numerical stability for RMSE to avoid nan values due to log(0)
#             for pred, out_path in zip(preds_resized, out_paths):
#                 np.save(out_path, pred.cpu())


num_epochs = 5
finetune_model(model, train_loader, val_loader, out_path=f"models/model_{run_id}_finetuned.pth", epochs=num_epochs)

# Reload the architecture
# model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
# model.to(device)

# Load the fine-tuned weights
# model.load_state_dict(torch.load("models/model_finetuned_final.pth", map_location=device))
model.eval()

print("✅ Loaded fine-tuned MiDaS model.")
#evaluate_model(model, val_loader, None, device)
visualize_prediction_with_ground_truth(model, val_loader, run_id, image_size, device, num_images=10)
#predict_model(model, test_loader)
visualize_prediction_without_ground_truth(model, test_loader, run_id,  image_size, device, num_images=10)