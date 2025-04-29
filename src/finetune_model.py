import os
import torch
import torch.nn as nn
from utils.dataloader import get_dataloaders
from utils.visualization import visualize_prediction_with_ground_truth, visualize_prediction_without_ground_truth
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

def torch_seed(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

torch_seed()

midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", skip_validation=True)
from midas.dpt_depth import DPT
from midas.blocks import Interpolate


class MiDaS_UQ(DPT):
    def __init__(self, path=None, non_negative=True, **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256
        head_features_1 = kwargs["head_features_1"] if "head_features_1" in kwargs else features
        head_features_2 = kwargs["head_features_2"] if "head_features_2" in kwargs else 32
        kwargs.pop("head_features_1", None)
        kwargs.pop("head_features_2", None)

        head = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 2, kernel_size=1, stride=1, padding=0),
            nn.Identity(),
        )
        super().__init__(head, **kwargs)
        self.relu = nn.ReLU(True) if non_negative else nn.Identity()
        self.softplus = nn.Softplus()
        
        if path is not None:
            self.load(path)

    def forward(self, x):
        output = super().forward(x)
        depth = self.relu(output[:, 0, :, :])
        logvar_depth = self.softplus(output[:, 1, :, :])
        return depth, logvar_depth
        
        # return super().forward(x)#.squeeze(dim=1)

class DepthUncertaintyLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.gaussian_nll = nn.GaussianNLLLoss(eps=eps)

    def forward(self, depth_pred, logvar_pred, depth_gt):
        # var = torch.exp(logvar_pred).clamp(min=1e-6)
        var = logvar_pred
        return self.gaussian_nll(depth_pred, depth_gt, var)  # + (1 / var).mean() * 1e-1


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

def scale_invariant_rmse(predicted, ground_truth):
    """
    predicted: Tensor of shape (B, 1, H, W)
    ground_truth: Tensor of shape (B, 1, H, W)
    Returns: scalar tensor (loss value)
    """

    # Flatten spatial dimensions
    B = predicted.size(0)
    predicted = predicted.reshape(B, -1)
    ground_truth = ground_truth.reshape(B, -1)

    # Log difference
    log_diff = torch.log(predicted) - torch.log(ground_truth)

    # Compute the global bias (alpha)
    alpha = torch.mean(log_diff, dim=1, keepdim=True)

    # Add bias and compute RMSE
    corrected_diff = log_diff + alpha  # Important! Add bias before squaring
    loss = torch.sqrt(torch.mean(corrected_diff ** 2, dim=1))

    return loss.mean() # scalar


def evaluate_model(model, val_loader, epoch):
    model.eval()
    total_rmse = 0.0
    with torch.no_grad():
        for images, depths in val_loader:
            images = images.to(device)
            depths = depths.to(device)
            
            depth, _ = model(images)
            depth_resized = torch.nn.functional.interpolate(
                depth.unsqueeze(1), size=depths.shape[-2:], mode="bicubic", align_corners=False
            )

            eps = 1e-8
            preds_resized = depth_resized.clamp(min=eps) # for numerical stability for RMSE to avoid nan values due to log(0)

            loss = scale_invariant_rmse(preds_resized, depths)
            total_rmse += loss.item()

    avg_rmse = total_rmse / len(val_loader)
    print(f"✅ Scale-Invariant RMSE after epoch {epoch+1}: {avg_rmse:.4f}")


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
#evaluate_model(model, val_loader, num_epochs)
visualize_prediction_with_ground_truth(model, val_loader, run_id, image_size, device, num_images=10)
#predict_model(model, test_loader)
visualize_prediction_without_ground_truth(model, test_loader, run_id,  image_size, device, num_images=10)