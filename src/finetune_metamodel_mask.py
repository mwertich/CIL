import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
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
from meta_models import SimpleUNet, AttentionUNet
from utils.expert_dataloader import ExpertTrainDataset, ExpertTestDataset
from utils.loss_funcs import scale_invariant_rmse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = [426, 560]


def entropy_loss(probs, mask=None, eps=1e-8):
    # probs: (B, num_experts, H, W)
    ent = -torch.sum(probs * torch.log(probs + eps), dim=1)  # shape: (B, H, W)
    
    if mask is not None:
        ent = ent * mask.float()  # Apply the spatial mask
        return ent.sum() / mask.float().sum()  # Mean over masked pixels
    else:
        return ent.mean()
    

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


def train_metamodel(model, train_dataloader, val_dataloader, categories, num_epochs=10, lr=1e-4, threshold=0.06, alpha=1, beta=0., gamma=0., tau=1, save_model=True):
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()


    #evaluate initial model
    evaluate_metamodel(model, val_dataloader, 0, categories, threshold, tau)

    model.train()
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        time = datetime.now()
        print(f"\n🕒 [{time.strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch}/{num_epochs}")

        # 🏄‍♂️ Wrap your dataloader with tqdm
        train_loader_tqdm = tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=True)

        for batch in train_loader_tqdm:
            images, depths, predictions, uncertainty = batch

            images = images.cuda().permute(0, 3, 1, 2)  # (B, C, H, W)
            depths = depths.cuda()  # (B, 1, H, W)
            uncertainty = uncertainty.cuda()  # (B, 1, H, W)

            expert_predictions = torch.stack(predictions, dim=0).squeeze(2).permute(1, 0, 2, 3).cuda()

            images_with_uncertainty = torch.cat((images, uncertainty), dim=1)

            logits = model(images_with_uncertainty) / tau  # (B, num_experts, H, W)
            logits = F.interpolate(logits, size=expert_predictions.shape[-2:], mode='bilinear', align_corners=False)

            # Prepare uncertainty and resize to match expert resolution
            uncertainty_min, uncertainty_max, uncertainty_mean, uncertainty_std = torch.min(uncertainty), torch.max(uncertainty), torch.mean(uncertainty), torch.std(uncertainty)
            uncertainty_mask = (uncertainty > threshold).squeeze(1)  # (B, H, W)

            # Get model prediction
            probs = F.softmax(logits, dim=1)  # (B, num_experts, H, W)
            pred_depth = torch.sum(probs * expert_predictions, dim=1, keepdim=True)  # (B, 1, H, W)

            # Model's argmax prediction (i.e., which expert it would choose)
            predicted_indices = torch.argmax(probs, dim=1)  # (B, H, W)

            errors = torch.abs(expert_predictions - depths.expand_as(expert_predictions))  # (B, num_experts, H, W)
            best_expert_indices = torch.argmin(errors, dim=1)  # (B, H, W)


            # Compute softmax-based depth only at uncertain pixels
            final_prediction = torch.empty_like(depths)

            # (1) Use expert 0 prediction where uncertainty is low
            base_model_prediction = expert_predictions[:, 0]
            final_prediction[:, 0][~uncertainty_mask] = base_model_prediction[~uncertainty_mask]

            # (2) Use soft combination where uncertainty is high
            soft_pred = torch.sum(probs * expert_predictions, dim=1)  # (B, H, W)
            final_prediction[:, 0][uncertainty_mask] = soft_pred[uncertainty_mask]

            # Compute per-pixel loss mask
            # Loss is computed where uncertainty is high (soft combination used)
            mask_loss = uncertainty_mask.unsqueeze(1)  # (B, 1, H, W)

            # Masked MSE Loss
            masked_final = final_prediction[mask_loss]
            masked_depths = depths[mask_loss]
            mse = mse_loss(masked_final, masked_depths)

            # Cross-Entropy Loss (masked)
            ce_raw = ce_loss_fn(logits, best_expert_indices)  # shape: scalar loss
            ce = (ce_raw * mask_loss.float()).mean()  # Apply mask

            # Final loss
            loss = alpha * mse + beta * ce + gamma * entropy_loss(probs, mask_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # ✏️ Update tqdm with current average loss
            train_loader_tqdm.set_postfix({'loss': running_loss / (train_loader_tqdm.n + 1e-8)})

        print(f"✅ Epoch [{epoch}/{num_epochs}] finished. Loss: {running_loss/len(train_dataloader):.4f}")
        evaluate_metamodel(model, val_dataloader, epoch, categories, threshold, tau)

    if save_model:
        torch.save(model.state_dict(), f"models/metamodel_final.pth")
        print(f"💾 Saved model checkpoint at models/metamodel_final.pth")
    return model


def evaluate_metamodel(model, val_dataloader, epoch, categories, threshold=0.03, tau=1., visualize=True):
    model.eval()
    total_rmse = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_dataloader:
            images, depths, predictions, uncertainty = batch
            images = images.cuda().permute(0, 3, 1, 2)  # (B, C, H, W)
            depths = depths.cuda()  # (B, 1, H, W)
            uncertainty = uncertainty.cuda()
            images_with_uncertainty = torch.cat((images, uncertainty), dim=1)

            # Stack expert predictions: (B, num_experts, 1, H, W)
            expert_predictions = torch.stack(predictions, dim=0).squeeze(2).permute(1, 0, 2, 3).cuda()

            logits = model(images_with_uncertainty) / tau  # (B, num_experts, H, W)
            logits = F.interpolate(logits, size=expert_predictions.shape[-2:], mode='bilinear', align_corners=False)
            # Softmax over experts
            probs = F.softmax(logits, dim=1)  # (B, num_experts, H, W)

            # Weighted sum of expert predictions
            pred_depths = torch.sum(probs * expert_predictions, dim=1, keepdim=True)  # (B, 1, H, W)

            # Model's argmax prediction (i.e., which expert it would choose)
            predicted_indices = torch.argmax(probs, dim=1)  # (B, H, W)

            errors = torch.abs(expert_predictions - depths.expand_as(expert_predictions))  # (B, num_experts, H, W)
            best_expert_indices = torch.argmin(errors, dim=1)  # (B, H, W)


            # Compute softmax-based depth only at uncertain pixels
            final_prediction = torch.empty_like(depths)

            
            uncertainty_mask = (uncertainty > threshold).squeeze(1)  # (B, H, W)

            # (1) Use expert 0 prediction where uncertainty is low
            base_model_prediction = expert_predictions[:, 0]
            final_prediction[:, 0][~uncertainty_mask] = base_model_prediction[~uncertainty_mask]
            predicted_indices[~uncertainty_mask] = 0

            # (2) Use soft combination where uncertainty is high
            soft_pred = torch.sum(probs * expert_predictions, dim=1)  # (B, H, W)
            final_prediction[:, 0][uncertainty_mask] = soft_pred[uncertainty_mask]


            loss = scale_invariant_rmse(final_prediction, depths)
            total_rmse += loss.item()


            # Expand GT depth to match experts shape
            depths_expanded = depths.expand_as(expert_predictions)  # (B, num_experts, H, W)

            # Calculate absolute error between each expert prediction and ground truth
            errors = torch.abs(expert_predictions - depths_expanded)  # (B, num_experts, H, W)

            # Find index of the expert with minimal error at each pixel
            best_expert_indices = torch.argmin(errors, dim=1)  # (B, H, W)

            if visualize and count == 0:
                visualize_batch(images, final_prediction, depths, probs, predicted_indices, best_expert_indices, uncertainty, uncertainty_mask, categories)
                count += 1

    avg_rmse = total_rmse / len(val_dataloader)
    print(f"✅ Scale-Invariant RMSE after epoch {epoch}: {avg_rmse:.4f}")


def predict_metamodel(model, test_dataloader, threshold=0.03, tau=1., prediction_folder="src/data/predictions"):
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            images, depth_file_names, predictions, uncertainty = batch
            images = images.cuda().permute(0, 3, 1, 2)  # (B, C, H, W)
            uncertainty = uncertainty.cuda()
            images_with_uncertainty = torch.cat((images, uncertainty), dim=1)

            # Stack expert predictions: (B, num_experts, 1, H, W)
            expert_predictions = torch.stack(predictions, dim=0).squeeze(2).permute(1, 0, 2, 3).cuda()

            logits = model(images_with_uncertainty) / tau  # (B, num_experts, H, W)
            logits = F.interpolate(logits, size=expert_predictions.shape[-2:], mode='bilinear', align_corners=False)
            # Softmax over experts
            probs = F.softmax(logits, dim=1)  # (B, num_experts, H, W)

            # Compute softmax-based depth only at uncertain pixels
            shape = list(images.shape)
            shape[1] = 1
            final_prediction = torch.empty(shape, dtype=images.dtype, device=images.device)

            uncertainty_mask = (uncertainty > threshold).squeeze(1)  # (B, H, W)

            # (1) Use expert 0 prediction where uncertainty is low
            base_model_prediction = expert_predictions[:, 0]
            final_prediction[:, 0][~uncertainty_mask] = base_model_prediction[~uncertainty_mask]

            # (2) Use soft combination where uncertainty is high
            soft_pred = torch.sum(probs * expert_predictions, dim=1)  # (B, H, W)
            final_prediction[:, 0][uncertainty_mask] = soft_pred[uncertainty_mask]

            for pred, depth_file_name in zip(final_prediction, depth_file_names):
                depth_file_path = os.path.join(prediction_folder, depth_file_name)
                np.save(depth_file_path, pred.cpu())





def visualize_batch(images, pred_depths, depths, probs_batch, predicted_indices_batch, best_expert_indices_batch, uncertainties, masks, categories, save_path="meta_maps"):
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
    for image, pred_depth, depth, probs, predicted_indices, best_expert_indices, mask, uncertainty in zip(images, pred_depths, depths, probs_batch, predicted_indices_batch, best_expert_indices_batch, masks, uncertainties):
        image = image.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        image = np.clip(image, 0, 1)

        pred_depth = pred_depth[0].cpu().numpy()
        depth_gt = depth[0].cpu().numpy()
        uncertainty = uncertainty[0].cpu().numpy()

        # Predicted expert map
        predicted_expert_map = predicted_indices.cpu().numpy()  # (H, W)
        mask = mask.cpu()


        # Best expert map (oracle)
        predicted_expert_map[~mask] = 6
        best_expert_indices[~mask] = 6
        
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
            [0, 0, 0],    # Black
        ]) / 255.0  # normalize to [0,1]

        colors = colors[:(num_models + 1)]
        num_colors = len(colors)

        expert_color_map = colors[predicted_expert_map]  # (H, W, 3)
        best_color_map = colors[best_map]      # (H, W, 3)

        # Plotting
        fig, (ax1, ax2) = plt.subplots(2, 6, figsize=(25, 8))  # ➡️ Now 11 plots

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
            expanded_mask = mask.cpu().numpy()[np.newaxis, :, :]
            masked_probs = probs[i].cpu().numpy() * expanded_mask # (H, W), values in [0, 1]
            valid_values = masked_probs[expanded_mask]
            color = colors[i]              # (3,), RGB in [0, 1]
            
            # Expand prob to (H, W, 1) to broadcast with (3,) color
            rgb_image = masked_probs.transpose(1, 2, 0) * color  # (H, W, 3)
            
            ax2[i].imshow(rgb_image)
            ax2[i].set_title(f'Prob {experts[i]}')
            ax2[i].text(0.5, -0.1, f'Mean: {np.mean(valid_values):.4f}, Std: {np.std(valid_values):.4f}, Max: {np.max(valid_values):.4f}, Min: {np.min(valid_values):.4f}', transform=ax2[i].transAxes,ha='center', va='top', fontsize=10)
            ax2[i].axis('off')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/sample{image_count}.png")
            print(f"✅ Saved visualization to {save_path}/sample{image_count}.png")
            image_count+=1
        else:
            plt.show()

        plt.close()


def main(config):

    root = "src/data"
    cluster_root = "src/data" # "/cluster/courses/cil/monocular_depth/data/"

    train_image_folder = os.path.join(cluster_root, "train")
    train_depth_folder = os.path.join(cluster_root, "train")
    val_image_folder = os.path.join(cluster_root, "train")
    val_depth_folder = os.path.join(cluster_root, "train")
    test_image_folder = os.path.join(cluster_root, "test")

    categories = ["kitchen", "bathroom", "dorm_room", "living_room", "home_office"]
    num_experts = len(categories) + 1
    base_predictions_path = os.path.join(root, "predictions_temp/base_model")
    expert_predictions_path = os.path.join(root, "predictions_temp/expert_models")

    train_image_depth_pairs = load_image_depth_pairs(os.path.join(root, config.train_list))
    val_image_depth_pairs = load_image_depth_pairs(os.path.join(root, config.val_list))
    test_image_depth_pairs = load_image_depth_pairs(os.path.join(root, "test_list.txt"))

    # Dataset and Dataloader
    train_dataset = ExpertTrainDataset(train_image_folder, train_depth_folder, train_image_depth_pairs, base_predictions_path, expert_predictions_path, categories)
    val_dataset = ExpertTrainDataset(val_image_folder, val_depth_folder, val_image_depth_pairs, base_predictions_path, expert_predictions_path, categories)
    test_dataset = ExpertTestDataset(test_image_folder, test_image_depth_pairs, base_predictions_path, expert_predictions_path, categories)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    
    model = AttentionUNet(in_channels=4, num_experts=num_experts)
    uncertainty_threshold = config.uncertainty_threshold
    tau=config.tau

    print("✅ Train Metamodel")
    train_metamodel(model, train_dataloader, val_dataloader, categories, num_epochs=config.num_epochs, threshold=uncertainty_threshold, tau=tau)
    print("Evaluate MetaModel")
    model_path = "models/metamodel_final.pth"
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()  # move to GPU after loading 
    model.eval()
    evaluate_metamodel(model, val_dataloader, config.num_epochs, categories, threshold=uncertainty_threshold, tau=tau)
    print("Predict MetaModel")
    #predict_metamodel(model, test_dataloader, threshold=uncertainty_threshold, tau=tau)
    print("Finished")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train metamodel to predict a pixel-wise linear combination of pixel from base and expert models")
    parser.add_argument("--train-list", type=str, required=True, help="Path to train list") # category_lists/bathroom_train_list.txt
    parser.add_argument("--val-list", type=str, required=True, help="Path to val list") # category_lists/bathroom_val_list.txt
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--uncertainty-threshold", type=int, default=0.05, help="Only evaluate loss at uncertain regions (uncertainty > threshold), otherwise base model")
    parser.add_argument("--tau", type=int, default=1., help="temperature of model outputs before softmax (logits)")
    config = parser.parse_args()

    run_id = datetime.now().strftime("%y%m%d_%H%M%S")
    print('---------------- Run id:', run_id, '----------------')

    main(config)
