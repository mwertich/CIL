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
import argparse
import imageio.v2 as imageio
import torch.nn.functional as F
import torchvision
from datetime import datetime
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = [426, 560]

class ExpertTrainDataset(Dataset):
    def __init__(self, image_dir, depth_dir, image_depth_file_pairs, base_predictions_path, expert_predictions_path, categories):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.image_depth_file_pairs = image_depth_file_pairs
        self.base_predictions_path = base_predictions_path
        self.expert_predictions_path = expert_predictions_path
        self.categories = categories

    def __len__(self):
        return len(self.image_depth_file_pairs)

    def __getitem__(self, idx):
        image_file_name, depth_file_name = self.image_depth_file_pairs[idx]
        image_path = os.path.join(self.image_dir, image_file_name)
        depth_path = os.path.join(self.depth_dir, depth_file_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path)

        image = torch.from_numpy(image / 255).float()
        depth = torch.from_numpy(depth).unsqueeze(0).float()
        predictions = []

        base_model_prediction_path = os.path.join(self.base_predictions_path, depth_file_name)
        base_pred_depth = torch.from_numpy(np.load(base_model_prediction_path)).float()
        
        uncertainty_path = os.path.join(self.base_predictions_path, depth_file_name.replace("depth", "uncertainty"))
        uncertainty = np.load(uncertainty_path)
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())

        predictions.append(base_pred_depth)

        for category in self.categories:
            expert_prediction_dir = os.path.join(self.expert_predictions_path, category)
            pred_file = os.path.join(expert_prediction_dir, depth_file_name)
            pred_depth = torch.from_numpy(np.load(pred_file)).float()
            predictions.append(pred_depth)

        return image, depth, predictions, uncertainty
    

class ExpertTestDataset(Dataset):
    def __init__(self, image_dir, image_depth_file_pairs, base_predictions_path, expert_predictions_path, categories):
        self.image_dir = image_dir
        self.image_depth_file_pairs = image_depth_file_pairs
        self.base_predictions_path = base_predictions_path
        self.expert_predictions_path = expert_predictions_path
        self.categories = categories

    def __len__(self):
        return len(self.image_depth_file_pairs)

    def __getitem__(self, idx):
        image_file_name, depth_file_name = self.image_depth_file_pairs[idx]
        image_path = os.path.join(self.image_dir, image_file_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = torch.from_numpy(image / 255).float()
        predictions = []

        base_model_prediction_path = os.path.join(self.base_predictions_path, depth_file_name)
        base_pred_depth = torch.from_numpy(np.load(base_model_prediction_path)).unsqueeze(0).float()

        uncertainty_path = os.path.join(self.base_predictions_path, depth_file_name.replace("depth", "uncertainty"))
        uncertainty = torch.from_numpy(np.load(uncertainty_path)).unsqueeze(0).float()
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())

        predictions.append(base_pred_depth)

        for category in self.categories:
            expert_prediction_dir = os.path.join(self.expert_predictions_path, category)
            pred_file = os.path.join(expert_prediction_dir, depth_file_name)
            pred_depth = torch.from_numpy(np.load(pred_file)).unsqueeze(0).float()
            predictions.append(pred_depth)
        return image, depth_file_name, predictions, uncertainty
    
    
# ===== Simple UNet =====
class SimpleUNet(nn.Module):
    def __init__(self, num_experts):
        super(SimpleUNet, self).__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())

        self.final = nn.Conv2d(64, num_experts, 1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')


    def center_crop(self, tensor, target_tensor):
        _, _, h, w = target_tensor.shape
        tensor = torchvision.transforms.functional.center_crop(tensor, [h, w])
        return tensor

    def forward(self, x):
        w_pad = (x.shape[-1] % 4) // 2
        h_pad = (x.shape[-2] % 4) // 2
        x_pad = F.pad(x, (w_pad, w_pad, h_pad, h_pad), mode='replicate')  # Pad to make it divisible by 4
        e1 = self.enc1(x_pad)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        b = self.bottleneck(p2)
        
        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        out = self.final(d1)
        return self.center_crop(out, x)


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

    return loss.mean()  # scalar


def entropy_loss(probs):
    # probs: (B, num_experts, H, W)
    return -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
    

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
            if len(parts) == 2:
                image_file, depth_file = parts
                pairs.append((image_file, depth_file))
    return pairs


def train_metamodel(model, train_dataloader, val_dataloader, categories, num_epochs=10, lr=1e-4, alpha=1, beta=0.001, tau=.5, save_model=True):
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()


    # evaluate initial model
    evaluate_metamodel(model, val_dataloader, 0, categories, tau)

    model.train()
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        time = datetime.now()
        print(f"\n🕒 [{time.strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch}/{num_epochs}")

        # 🏄‍♂️ Wrap your dataloader with tqdm
        train_loader_tqdm = tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=True)

        for batch in train_loader_tqdm:
            images, depths, predictions, uncertainties = batch

            images = images.cuda().permute(0, 3, 1, 2)  # (B, C, H, W)
            depths = depths.cuda()  # (B, 1, H, W)

            experts = torch.stack(predictions, dim=0).squeeze(2).permute(1, 0, 2, 3).cuda()

            logits = model(images) / tau  # (B, num_experts, H, W) (tau = temperature for softmax and cross entropy loss)

            probs = F.softmax(logits, dim=1)  # (B, num_experts, H, W)
            pred_depth = torch.sum(probs * experts, dim=1, keepdim=True)  # (B, 1, H, W)

            errors = torch.abs(experts - depths.expand_as(experts))  # (B, num_experts, H, W)
            best_expert_indices = torch.argmin(errors, dim=1)  # (B, H, W)

            mse_loss_term = mse_loss(pred_depth, depths)
            ce_loss_term = ce_loss(logits, best_expert_indices)
            entropy_loss_term = entropy_loss(probs)

            loss = mse_loss_term + alpha * ce_loss_term + beta * entropy_loss_term

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # ✏️ Update tqdm with current average loss
            train_loader_tqdm.set_postfix({'loss': running_loss / (train_loader_tqdm.n + 1e-8)})

        print(f"✅ Epoch [{epoch}/{num_epochs}] finished. Loss: {running_loss/len(train_dataloader):.4f}")
        evaluate_metamodel(model, val_dataloader, epoch, categories, tau)

    if save_model:
        torch.save(model.state_dict(), f"models/metamodel_final.pth")
        print(f"💾 Saved model checkpoint at models/metamodel_final.pth")
    return model


def evaluate_metamodel(model, val_dataloader, epoch, categories, tau=1., visualize=True):
    model.eval()
    total_rmse = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_dataloader:
            images, depths, predictions, uncertainties = batch
            images = images.cuda().permute(0, 3, 1, 2)  # (B, C, H, W)
            depths = depths.cuda()  # (B, 1, H, W)

            # Stack expert predictions: (B, num_experts, 1, H, W)
            experts = torch.cat(predictions, dim=1).cuda()

            logits = model(images) / tau  # (B, num_experts, H, W)
            # Softmax over experts
            probs = F.softmax(logits, dim=1)  # (B, num_experts, H, W)

            # Weighted sum of expert predictions
            pred_depths = torch.sum(probs * experts, dim=1, keepdim=True)  # (B, 1, H, W)

            loss = scale_invariant_rmse(pred_depths, depths)
            total_rmse += loss.item()


            # Expand GT depth to match experts shape
            depths_expanded = depths.expand_as(experts)  # (B, num_experts, H, W)

            # Calculate absolute error between each expert prediction and ground truth
            errors = torch.abs(experts - depths_expanded)  # (B, num_experts, H, W)

            # Find index of the expert with minimal error at each pixel
            best_expert_indices = torch.argmin(errors, dim=1)  # (B, H, W)

            if visualize and count == 0:
                visualize_batch(images, pred_depths, depths, probs, best_expert_indices, uncertainties, categories)
                count += 1

    avg_rmse = total_rmse / len(val_dataloader)
    print(f"✅ Scale-Invariant RMSE after epoch {epoch}: {avg_rmse:.4f}")


def visualize_batch(images, pred_depths, depths, probs_batch, best_expert_indices_batch, uncertainties, categories, save_path="meta_maps"):
    """
    Visualize a batch of examples.

    Args:
        images (Tensor): (B, 3, H, W)
        pred_depths (Tensor): (B, 1, H, W)
        depths (Tensor): (B, 1, H, W)
        probs (Tensor): (B, num_experts, H, W)
        best_expert_indices (Tensor): (B, H, W)
        uncertainties (Tensor) (B, 1, H, W)
        save_path (str or None): If given, saves the figure instead of showing.
    """
    batch_size = images.shape[0]
    num_models = probs_batch.shape[1]

    # Create save directory if needed
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    
    image_count = 0
    for image, pred_depth, depth, probs, best_expert_indices, uncertainty in zip(images, pred_depths, depths, probs_batch, best_expert_indices_batch, uncertainties):
        image = image.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        image = np.clip(image, 0, 1)

        pred_depth = pred_depth[0].cpu().numpy()
        depth_gt = depth[0].cpu().numpy()
        uncertainty = uncertainty[0].cpu().numpy()

        # Predicted expert map
        expert_map = torch.argmax(probs, dim=0).cpu().numpy()  # (H, W)

        probs_mean = torch.mean(probs, dim=0).cpu().numpy()
        probs_var = torch.var(probs, dim=0).cpu().numpy()

        # Best expert map (oracle)
        best_map = best_expert_indices.cpu().numpy()  # (H, W)

        experts = ["base"] + categories 

        # Define a color map: one color per expert
        colors = np.array([
            [255, 0, 0],      # Red
            [0, 255, 0],      # Green
            [0, 0, 255],      # Blue
            [255, 255, 0],    # Yellow
            [255, 0, 255],    # Magenta
            [0, 255, 255],    # Cyan
        ]) / 255.0  # normalize to [0,1]

        colors = colors[:num_models]
        num_colors = len(colors)

        expert_color_map = colors[expert_map]  # (H, W, 3)
        best_color_map = colors[best_map]      # (H, W, 3)

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 6, figsize=(25, 10))  # ➡️ Now 11 plots

        ax1[0].imshow(image)
        ax1[0].set_title('Input Image')

        ax1[1].imshow(pred_depth, cmap='plasma')
        ax1[1].set_title('Predicted Depth')

        ax1[2].imshow(depth_gt, cmap='plasma')
        ax1[2].set_title('Ground Truth Depth')

        ax1[3].imshow(expert_color_map)
        ax1[3].set_title('Chosen Expert Map')

        ax1[4].imshow(best_color_map)
        ax1[4].set_title('Best Expert (Oracle) Map')

        ax1[5].imshow(uncertainty, cmap='viridis')
        ax1[5].set_title('Uncertainty Map')

        for i in range(6):
            ax1[i].axis('off')

        for i in range(num_models):
            prob = probs[i].cpu().numpy()  # (H, W), values in [0, 1]
            color = colors[i]              # (3,), RGB in [0, 1]
            
            # Expand prob to (H, W, 1) to broadcast with (3,) color
            rgb_image = prob[..., np.newaxis] * color  # (H, W, 3)
            
            ax2[i].imshow(rgb_image)
            ax2[i].set_title(f'Prob {experts[i]}')
            ax2[i].text(0.5, -0.1, f'Max: {np.max(prob):.4f}, Min: {np.min(prob):.4f}, Mean: {np.mean(prob):.4f}, Std: {np.std(prob):.4f}', transform=ax2[i].transAxes,ha='center', va='top', fontsize=10)
            ax2[i].axis('off')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/sample{image_count}.png")
            print(f"✅ Saved visualization to {save_path}/sample{image_count}.png")
            image_count+=1
        else:
            plt.show()

        plt.close()


def main(args):

    root = "src/data"
    cluster_root = "src/data" # "/cluster/courses/cil/monocular_depth/data/"

    train_image_folder = os.path.join(cluster_root, "train")
    train_depth_folder = os.path.join(cluster_root, "train")
    test_image_folder = os.path.join(cluster_root, "test")

    categories = ["kitchen", "bathroom", "dorm_room", "living_room", "home_office"]
    num_experts = len(categories) + 1
    base_predictions_path = os.path.join(root, "predictions_temp/base_model")
    expert_predictions_path = os.path.join(root, "predictions_temp/expert_models")

    train_image_depth_pairs = load_image_depth_pairs(os.path.join(root, args.train_list))
    val_image_depth_pairs = load_image_depth_pairs(os.path.join(root, args.val_list))
    #test_image_depth_pairs = load_image_depth_pairs(os.path.join(root, args.test_list))

    # Dataset and Dataloader
    train_dataset = ExpertTrainDataset(train_image_folder, train_depth_folder, train_image_depth_pairs, base_predictions_path, expert_predictions_path, categories)
    val_dataset = ExpertTrainDataset(train_image_folder, train_depth_folder, val_image_depth_pairs, base_predictions_path, expert_predictions_path, categories)
    test_dataset = ExpertTestDataset(train_image_folder, train_image_depth_pairs, base_predictions_path, expert_predictions_path, categories)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    
    model = SimpleUNet(num_experts=num_experts)

    print("✅ Train Metamodel")
    train_metamodel(model, train_dataloader, val_dataloader, categories, num_epochs=args.num_epochs)
    print("Evaluate MetaModel")
    model_path = "models/metamodel_final.pth"
    #model.load_state_dict(torch.load(model_path))
    model = model.cuda()  # move to GPU after loading 
    model.eval()
    evaluate_metamodel(model, val_dataloader, 5, categories, visualize=True)
    print("Finished")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train metamodel to predict a pixel-wise linear combination of pixel from base and expert models")
    parser.add_argument("--train-list", type=str, required=True, help="Path to train list") # category_lists/bathroom_train_list.txt
    parser.add_argument("--val-list", type=str, required=True, help="Path to val list") # category_lists/bathroom_val_list.txt
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=5)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%y%m%d_%H%M%S")
    print('---------------- Run id:', run_id, '----------------')

    main(args)