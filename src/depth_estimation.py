import os
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import cv2
import matplotlib.cm as cm
from PIL import Image

# --- Setup paths ---
image_folder = "src/data/train"      # <-- CHANGE THIS
output_folder = "src/depth_npy/train_predict"
os.makedirs(output_folder, exist_ok=True)

# --- Load model ---
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large") 
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

# Load the fine-tuned weights
midas.load_state_dict(torch.load("models/model_finetuned.pth", map_location=device))
midas.eval()

midas.to(device)

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

    inverse = False
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

    # Apply colormap
    cmap = cm.get_cmap("plasma")
    colored_depth = cmap(depth)[:, :, :3]  # Drop alpha channel

    # Convert to uint8 RGB
    depth_img = (colored_depth * 255).astype(np.uint8)
    Image.fromarray(depth_img).save(output_path)


npy_map_folder = "src/depth_npy/train_predict" 
depth_map_folder = "src/depth_maps/train_predict"
npy_files = [f for f in os.listdir(npy_map_folder) if f.lower().endswith(tuple([".npy"]))]
npy_files = npy_files


for npy_file in tqdm(npy_files, desc="Saving depths maps as .png"):
    npy_path = os.path.join(npy_map_folder, npy_file)
    base_name = os.path.splitext(npy_file)[0]
    out_path = os.path.join(depth_map_folder, f"{base_name}.png")
    save_depth_as_image(npy_path, output_path=out_path)
