import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF


from midas.dpt_depth import DPT

class MiDaS_UQ(DPT):
    def __init__(self, **kwargs):
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
                # nn.ReLU(True) if non_negative else nn.Identity(),
                # nn.Identity(),
            )
    
            super().__init__(head, **kwargs)
    
            if path is not None:
               self.load(path)

        def forward(self, x):
            output = super().forward(x)#.squeeze(dim=1)
            depth = output[:, 0, :, :]#.squeeze(dim=1)
            logvar_depth = output[:, 1, :, :]#.squeeze(dim=1)
            return depth, logvar_depth
            
            # return super().forward(x)#.squeeze(dim=1)

class DepthUncertaintyLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.gaussian_nll = nn.GaussianNLLLoss(eps=eps)

    def forward(self, depth_pred, logvar_depth, depth_gt):
        var = torch.exp(logvar_pred).clamp(min=1e-6)
        return self.gaussian_nll(depth_pred, depth_gt, var)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageDepthDataset(Dataset):
    def __init__(self, image_dir, depth_dir, transform, image_depth_file_pairs):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.image_depth_file_pairs = image_depth_file_pairs

    def __len__(self):
        return len(self.image_depth_file_pairs)

    def __getitem__(self, idx):
        image_file, depth_file = self.image_depth_file_pairs[idx]
        image_path = os.path.join(self.image_dir, image_file)
        depth_path = os.path.join(self.depth_dir, depth_file)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path)

        image = self.transform(image).squeeze(0)
        depth = torch.from_numpy(depth).unsqueeze(0).float()

        return image, depth
    

class TestImageDepthDataset(Dataset):
    def __init__(self, image_dir, depth_dir, transform, image_depth_file_pairs):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.image_depth_file_pairs = image_depth_file_pairs

    def __len__(self):
        return len(self.image_depth_file_pairs)

    def __getitem__(self, idx):
        image_file, depth_file = self.image_depth_file_pairs[idx]
        image_path = os.path.join(self.image_dir, image_file)
        depth_path = os.path.join(self.depth_dir, depth_file)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image).squeeze(0)
        return image, depth_path


def load_image_depth_pairs(file_path):
    """
    Reads a text file containing pairs of image and depth file names.
    Args:
        file_path (str): Path to the text file.
    Returns:
        List[Tuple[str, str]]: List of (image_filename, depth_filename) pairs.
    """
    pairs = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            image_file, depth_file = parts
            pairs.append((image_file, depth_file))
    return pairs


# Load model and transforms
# model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
model = MiDaS_UQ(path='/home/evrachoriti/.cache/torch/hub/checkpoints/dpt_large_384.pt')
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

image_size = [426, 560]

train_image_folder = "src/data/train"
train_depth_folder = "src/data/train"
test_image_folder = "src/data/test"
test_depth_folder = "src/data/predictions"

#all_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]
train_image_depth_pairs = load_image_depth_pairs("src/data/train_list.txt")
test_image_depth_pairs = load_image_depth_pairs("src/data/test_list.txt")

subset_size = 1000 #23971
val_size = 0.1

files = train_image_depth_pairs[:subset_size] if subset_size else train_image_depth_pairs
train_pairs, val_pairs = train_test_split(files, test_size=val_size, random_state=42)
test_pairs = test_image_depth_pairs


# Dataset and Dataloader
train_batch_size, val_batch_size, test_batch_size = 1, 1, 1

train_dataset = ImageDepthDataset(train_image_folder, train_depth_folder, transform, train_pairs)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

val_dataset = ImageDepthDataset(train_image_folder, train_depth_folder, transform, val_pairs)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

test_dataset = TestImageDepthDataset(test_image_folder, test_depth_folder, transform, test_pairs)
test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=True)



def scale_invariant_rmse_new(predicted, ground_truth):
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

    # Compute scale-invariant RMSE
    squared_mean = torch.mean(log_diff ** 2, dim=1)
    mean_squared = torch.mean(log_diff, dim=1) ** 2
    loss = torch.sqrt(squared_mean - mean_squared)
    return loss.mean()  # scalar


def scale_invariant_rmse(predicted, ground_truth):
    """
    predicted: Tensor of shape (B, 1, H, W)
    ground_truth: Tensor of shape (B, 1, H, W)
    Returns: scalar tensor (loss value)
    """

    # Scale-Invariant RMSE Loss with NaN protection.

    # Flatten to (B, H*W)
    predicted = predicted.view(predicted.size(0), -1)
    ground_truth = ground_truth.view(ground_truth.size(0), -1)

    # Log transform
    log_pred = torch.log(predicted)
    log_gt = torch.log(ground_truth)

    # Difference
    d = log_pred - log_gt  # shape: (B, N)
    n = d.shape[1]

    # Alpha (per image in batch)
    alpha = torch.mean(log_gt - log_pred, dim=1, keepdim=True)  # shape: (B, 1)

    # Final loss
    loss = torch.sqrt(torch.mean((d + alpha) ** 2, dim=1))  # shape: (B,)
    return loss.mean()  # scalar


def evaluate_model(model, val_loader, epoch):
    model.eval()
    total_rmse = 0.0
    with torch.no_grad():
        for images, depths in val_loader:
            images = images.to(device)
            depths = depths.to(device)

            preds = model(images)
            preds_resized = torch.nn.functional.interpolate(
                preds.unsqueeze(1), size=depths.shape[-2:], mode="bicubic", align_corners=False
            )

            eps = 1e-8
            preds_resized = preds_resized.clamp(min=eps) # for numerical stability for RMSE to avoid nan values due to log(0)

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
            # outputs = model(images)
            depth, logvar_depth = model(images)

            depth_resized = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=depths.shape[-2:],
                mode="bicubic",
                align_corners=False
            ).squeeze(1)

            # outputs_resized = torch.nn.functional.interpolate(
            #     outputs.unsqueeze(1),
            #     size=depths.shape[-2:],
            #     mode="bicubic",
            #     align_corners=False
            # ).squeeze(1)

            loss = criterion(depth_resized, logvar_depth, depths.squeeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
        evaluate_model(model, val_loader, epoch)
        # Save model after each epoch
        
        model_path = f"models/model_finetuned_epoch_{epoch}.pth"
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

#             preds = model(images)
#             preds_resized = torch.nn.functional.interpolate(
#                 preds.unsqueeze(1), size=image_size, mode="bicubic", align_corners=False
#             )
#             eps = 1e-8
#             preds_resized = preds_resized.clamp(min=eps) # for numerical stability for RMSE to avoid nan values due to log(0)
#             for pred, out_path in zip(preds_resized, out_paths):
#                 np.save(out_path, pred.cpu())



def denormalize_image(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    return tensor.clamp(0, 1)



def visualize_depth_maps(plt_title, file_name, image, prediction, ground_truth=None):
    # Convert tensors to numpy for visualization
    img_vis = denormalize_image(image.squeeze(0).cpu())
    img_np = TF.to_pil_image(img_vis)
    pred_np = prediction.squeeze().cpu().numpy()
    # Normalize depth maps for display
    pred_disp = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min())

    if ground_truth is not None:
        depth_np = ground_truth.squeeze().cpu().numpy()
        gt_disp = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
        captions = ["Input Image", "Ground Truth", "Predicted Depth"]
        plot_images = [img_np, gt_disp, pred_disp]
    else:
        captions = ["Input Image", "Predicted Depth"]
        plot_images = [img_np, pred_disp]

    
    # Plot

    fig, axs = plt.subplots(1, len(captions), figsize=(12, 4))
    for (i, (caption, plot_img)) in enumerate(zip(captions, plot_images)):
        axs[i].set_title(caption)
        if i == 0:
            axs[i].imshow(plot_img)
        else:
            axs[i].imshow(plot_img, cmap="plasma")

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig(file_name)



def visualize_prediction_with_ground_truth(model, loader, num_images=5):
    model.eval()

    images_shown = 0
    with torch.no_grad():
        for images, depths in loader:
            images = images.to(device)
            depths = depths.to(device)

            depth, logvar_depth = model(images)
            depth_resized = torch.nn.functional.interpolate(
                depth.unsqueeze(1), size=depths.shape[-2:], mode="bicubic", align_corners=False
            )
            
            # pred = model(images)
            # pred_resized = torch.nn.functional.interpolate(
            #     pred.unsqueeze(1), size=depths.shape[-2:], mode="bicubic", align_corners=False
            # )
            for image, dep, logvar_dep, prediction in zip(images, depths, logvar_depth, depth_resized): 
                images_shown += 1
                file_name = f"depth_maps/val/depth_map_{images_shown}.png"
                visualize_depth_maps("Depths Map Validation Set", file_name, image, prediction, dep)
                file_name = f"depth_maps/val/uq_map_{images_shown}.png"
                visualize_depth_maps("Uncertainty Map Validation Set", file_name, image, prediction, logvar_dep)
                if images_shown >= num_images:
                    return


def visualize_prediction_without_ground_truth(model, test_loader, num_images=5):
    model.eval()

    images_shown = 0
    with torch.no_grad():
        for images, out_paths in test_loader:

            images = images.to(device)

            depth, logvar_depth = model(images)
            depth_resized = torch.nn.functional.interpolate(
                depth.unsqueeze(1), size=depths.shape[-2:], mode="bicubic", align_corners=False
            )
            
            # pred = model(images)
            # pred_resized = torch.nn.functional.interpolate(
            #     pred.unsqueeze(1), size=image_size, mode="bicubic", align_corners=False
            # )
            for image, logvar_dep, prediction in zip(images, logvar_depth, depth_resized): 
                images_shown += 1
                file_name = f"depth_maps/val/depth_map_{images_shown}.png"
                visualize_depth_maps("Depths Map Validation Set", file_name, image, prediction)
                file_name = f"depth_maps/val/uq_map_{images_shown}.png"
                visualize_depth_maps("Uncertainty Map Validation Set", file_name, image, prediction, logvar_dep)
                if images_shown >= num_images:
                    return


num_epochs = 1
finetune_model(model, train_loader, val_loader, out_path="models/model_finetuned.pth", epochs=num_epochs)

# Reload the architecture
model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
model.to(device)

# Load the fine-tuned weights
model.load_state_dict(torch.load("models/model_finetuned_final.pth", map_location=device))
model.eval()

print("✅ Loaded fine-tuned MiDaS model.")
#evaluate_model(model, val_loader, num_epochs)
visualize_prediction_with_ground_truth(model, val_loader, num_images=10)
#predict_model(model, test_loader)
visualize_prediction_without_ground_truth(model, test_loader, num_images=10)