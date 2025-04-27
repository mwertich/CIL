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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = [426, 560]

class ImageDepthDataset(Dataset):
    def __init__(self, image_dir, transform, image_depth_file_pairs):
        self.image_dir = image_dir
        self.transform = transform
        self.image_depth_file_pairs = image_depth_file_pairs

    def __len__(self):
        return len(self.image_depth_file_pairs)

    def __getitem__(self, idx):
        image_file, _ = self.image_depth_file_pairs[idx]
        image_path = os.path.join(self.image_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image).squeeze(0)

        return image, image_file
    


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




def predict_ensemble(models, categories, loader, out_dir, eps=1e-8):
    for model, category in zip(models, categories):
        # Reload the architecture
        model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        model.to(device)
        # Load the fine-tuned weights
        model_path = f"models/model_{category}_finetuned.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ Loaded fine-tuned MiDaS model for category {category}.")
        out_path = os.path.join(out_dir, category)
        with torch.no_grad():
            for images, image_file_names in loader:
                images = images.to(device)
                preds = model(images)
                preds_resized = torch.nn.functional.interpolate(
                    preds.unsqueeze(1), size=image_size, mode="bicubic", align_corners=False
                )
                preds_resized = preds_resized.clamp(min=eps) # for numerical stability for RMSE to avoid nan values due to log(0)
                for pred, file_name in zip(preds_resized, image_file_names):
                    storage_path = os.path.join(out_path, f"{file_name[:-8]}_depth")
                    np.save(storage_path, pred.cpu())


def load_predictions(categories, out_dir, visualize=False):
    for category in categories:
        category_dir = os.path.join(out_dir, category)
        if not os.path.exists(category_dir):
            print(f"⚠️ Category folder '{category_dir}' does not exist. Skipping.")
            continue

        # List all .npy files
        all_files = [f for f in os.listdir(category_dir) if f.endswith('.npy')]

        if not all_files:
            print(f"⚠️ No .npy files found in {category_dir}. Skipping.")
            continue
        
        if visualize:
            selected_files = all_files[:2]
            for selected_file in selected_files:
                path = os.path.join(category_dir, selected_file)
                prediction = np.load(path)
                visualize_depth_map(selected_file, prediction)



def predict_ensemble_16_bit(models, categories, loader, out_dir, eps=1e-8):
    for model, category in zip(models, categories):
        # Reload the architecture
        model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        model.to(device)
        # Load the fine-tuned weights
        model_path = f"models/model_{category}_finetuned.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ Loaded fine-tuned MiDaS model for category {category}.")
        out_path = os.path.join(out_dir, category)
        os.makedirs(out_path, exist_ok=True)

        with torch.no_grad():
            for images, image_file_names in loader:
                images = images.to(device)
                preds = model(images)
                preds_resized = torch.nn.functional.interpolate(
                    preds.unsqueeze(1), size=image_size, mode="bicubic", align_corners=False
                )
                preds_resized = preds_resized.clamp(min=eps)

                for pred, file_name in zip(preds_resized, image_file_names):
                    pred_cpu = pred.squeeze(0).cpu().numpy()

                    # Save original min and max
                    min_val = pred_cpu.min()
                    max_val = pred_cpu.max()

                    # Normalize to [0, 1]
                    pred_norm = (pred_cpu - min_val) / (max_val - min_val + eps)

                    # Scale to uint16
                    pred_uint16 = (pred_norm * 65535).astype(np.uint16)

                    # Save PNG
                    storage_path = os.path.join(out_path, f"{file_name[:-8]}_depth.png")
                    imageio.imwrite(storage_path, pred_uint16)

                    # Save min and max separately
                    np.save(storage_path.replace('_depth.png', '_minmax.npy'), np.array([min_val, max_val]))


def load_predictions_16_bit(categories, out_dir, visualize=True):
    for category in categories:
        category_dir = os.path.join(out_dir, category)
        if not os.path.exists(category_dir):
            print(f"⚠️ Category folder '{category_dir}' does not exist. Skipping.")
            continue

        all_files = [f for f in os.listdir(category_dir) if f.endswith('_depth.png')]

        if not all_files:
            print(f"⚠️ No .png files found in {category_dir}. Skipping.")
            continue

        if visualize:
            selected_files = all_files[:3]
            for selected_file in selected_files:
                path = os.path.join(category_dir, selected_file)
                minmax_path = path.replace('_depth.png', '_minmax.npy')

                # Load prediction and min/max
                pred_uint16 = imageio.imread(path)
                min_val, max_val = np.load(minmax_path)

                # Recover normalized float
                pred_norm = pred_uint16.astype(np.float32) / 65535.0

                # Recover original scale (pred_float32 is now the full restored 32-bit depth map 🎯)
                pred_float32 = pred_norm * (max_val - min_val) + min_val

                visualize_depth_map(selected_file, pred_float32)


def visualize_depth_map(file_name, prediction):
    # Convert tensors to numpy for visualization
    pred_np = prediction.squeeze()
    # Normalize depth maps for display
    pred_disp = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min())
    # Plot
    plt.figure(figsize=(6, 4))
    plt.imshow(pred_disp, cmap="plasma")
    plt.tight_layout()
    plt.show()
    print(f"Save {file_name}")
    plt.savefig(f"{file_name[:-4]}.png")




def main(args):
    # Load transforms
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    root = "src/data"
    cluster_root = "src/data" # "/cluster/courses/cil/monocular_depth/data/"

    train_image_folder = os.path.join(cluster_root, "train")
    test_image_folder = os.path.join(cluster_root, "test")

    categories = ["kitchen", "bathroom", "dorm_room", "living_room", "home_office"][:1]

    expert_models = [f"models/model_{category}_finetuned.pth" for category in categories]

    image_depth_pairs = load_image_depth_pairs(os.path.join(root, args.sample_list))[:3]


    # Dataset and Dataloader
    dataset = ImageDepthDataset(train_image_folder, transform, image_depth_pairs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    out_dir = "src/data/predictions/expert_models"

    print("✅ Predict with fine-tuned MiDaS model ensemble.")
    predict_ensemble_16_bit(expert_models, categories, dataloader, out_dir)
    print("✅ Load predictions from  MiDaS model ensemble.")
    load_predictions_16_bit(categories, out_dir)
    print("Finished")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune MiDaS model for indoor depth estimation by category.")
    parser.add_argument("--sample-list", type=str, required=True, help="Path to sample list")
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args()

    main(args)