import os
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import cv2

# --- Setup paths ---
image_folder = "src/data/train/train"      # <-- CHANGE THIS
output_folder = "src/depth_npy/train_predict"
npy_map_folder = "src/depth_npy/train_predict" 
depth_map_folder = "src/depth_maps/train_predict"
os.makedirs(output_folder, exist_ok=True)

# --- Load model ---
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")  # or "DPT_Hybrid"
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

transform = midas_transforms.dpt_transform

# --- Iterate through all images ---
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(tuple([".png"]))]
image_files = image_files

for image_file in tqdm(image_files, desc="Saving depth as .npy"):
    img_path = os.path.join(image_folder, image_file)
    #img = Image.open(img_path).convert("RGB")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Transform and send to device
    input_tensor = transform(img).to(device)

    # Run depth estimation
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(img.shape[0], img.shape[1]),  # Resize to original image size
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Convert to numpy
    depth_map = prediction.cpu().numpy()

    inverse = True
    if inverse:
        depth_map = 1.0 / (depth_map + 1e-8)

    # Save as .npy
    base_name = os.path.splitext(image_file)[0]
    npy_path = os.path.join(output_folder, f"{base_name}.npy")
    np.save(npy_path, depth_map)

print("✅ All depth maps saved as .npy files in:", output_folder)


def save_depth_as_image(npy_path, output_path, normalize=True, invert=False):
    """
    Converts a .npy depth map to a grayscale image and saves it.

    Args:
        npy_path (str): Path to the .npy file.
        output_path (str): Where to save the output .png image.
        normalize (bool): Whether to normalize the depth for visualization.
    """
    depth = np.load(npy_path)

    if normalize:
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 0:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = np.zeros_like(depth)

    depth_img = (depth * 255).astype(np.uint8)
    Image.fromarray(depth_img).save(output_path)



npy_files = [f for f in os.listdir(npy_map_folder) if f.lower().endswith(tuple([".npy"]))]
npy_files = npy_files

for npy_file in tqdm(npy_files, desc="Saving depths maps as .png"):
    npy_path = os.path.join(npy_map_folder, npy_file)
    base_name = os.path.splitext(npy_file)[0]
    out_path = os.path.join(depth_map_folder, f"{base_name}.png")
    save_depth_as_image(npy_path, output_path=out_path)


def scale_invariant_rmse(pred_depth, gt_depth, mask=None, eps=1e-8):
    """
    pred_depth: np.ndarray of predicted depth map
    gt_depth: np.ndarray of ground truth depth map
    mask: optional binary mask to include only valid pixels (e.g. gt_depth > 0)
    eps: small value to avoid log(0)
    """
    # Flatten and mask
    if mask is None:
        mask = (gt_depth > 0) & (pred_depth > 0)

    pred = pred_depth[mask].astype(np.float32)
    gt = gt_depth[mask].astype(np.float32)

    # Apply log transform
    log_pred = np.log(pred + eps)
    log_gt = np.log(gt + eps)

    delta = log_pred - log_gt
    alpha = np.mean(log_gt - log_pred)

    si_rmse = np.sqrt(np.mean((delta + alpha) ** 2))
    return si_rmse



npy_map_folder_train = "src/data/train/train" 
npy_map_folder_predict = "src/depth_npy/train_predict"
rmse_scores = []

def compare_depth_maps(npy_map_folder_train, npy_map_folder_predict):
    predictions = [f for f in os.listdir(npy_map_folder_train) if f.lower().endswith(tuple([".npy"]))][:100]
    ground_truths = [f for f in os.listdir(npy_map_folder_predict) if f.lower().endswith(tuple([".npy"]))][:100]
    for (predicted_file, ground_truth_file) in zip(predictions, ground_truths):
        predicted_path = os.path.join(npy_map_folder_train, predicted_file)
        ground_truth_path = os.path.join(npy_map_folder_predict, ground_truth_file)
        pred = np.load(predicted_path)
        gt = np.load(ground_truth_path)
        si_rmse = scale_invariant_rmse(pred, gt)
        rmse_scores.append(si_rmse)
        print("Scale-Invariant RMSE:", si_rmse)


compare_depth_maps(npy_map_folder_train, npy_map_folder_predict)
print("Scale-Invariant RMSE Mean:", np.mean(rmse_scores))
print("Scale-Invariant RMSE: Std", np.std(rmse_scores))