import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler
import numpy as np
import cv2


# Sharpening function
def augment_image(image, type):
    if type not in ['smooth', 'sharpen', 'none']:
        raise ValueError("Invalid type. Expected 'smooth', 'sharpen' or 'none'.")
    if type == 'none':
        return image

    if type == 'smooth':
        # Apply Gaussian blur (you can tweak the kernel size and sigma)
        blurred_image = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=1.0)

        # Convert it back to RGB
        smoothed_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
        return smoothed_image
    else:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

        # Convert image to BGR since OpenCV uses BGR format
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Apply sharpening filter
        sharpened_image = cv2.filter2D(image_bgr, -1, kernel)

        # Convert it back to RGB
        sharpened_image = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)
        return sharpened_image


class ImageDepthDataset(Dataset):
    def __init__(self, image_dir, depth_dir, transform, image_depth_file_pairs, augmentation='none'):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.image_depth_file_pairs = image_depth_file_pairs
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_depth_file_pairs)

    def __getitem__(self, idx):
        image_file, depth_file = self.image_depth_file_pairs[idx]
        image_path = os.path.join(self.image_dir, image_file)
        depth_path = os.path.join(self.depth_dir, depth_file)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path)

        image = augment_image(image, self.augmentation)  # Apply sharpening

        image = self.transform(image).squeeze(0)
        depth = torch.from_numpy(depth).unsqueeze(0).float()

        return image, depth
    

class TestImageDepthDataset(Dataset):
    def __init__(self, image_dir, depth_dir, transform, image_depth_file_pairs, augmentation='none'):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.image_depth_file_pairs = image_depth_file_pairs
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_depth_file_pairs)

    def __getitem__(self, idx):
        image_file, depth_file = self.image_depth_file_pairs[idx]
        image_path = os.path.join(self.image_dir, image_file)
        depth_path = os.path.join(self.depth_dir, depth_file)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = augment_image(image, self.augmentation)  # Apply sharpening
            
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
            if len(parts):
                image_file, depth_file = parts
                pairs.append((image_file, depth_file))
    return pairs



def get_dataloaders(image_size, train_size, val_size, batch_size, train_list="train_list.txt", val_list="val_list.txt", test_list="train_list.txt", augmentation='none'):

    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    
    root = "src/data"

    train_image_folder = os.path.join(root, "train")
    train_depth_folder = os.path.join(root, "train")
    test_image_folder = os.path.join(root, "test")
    test_depth_folder = os.path.join(root, "predictions")

    train_list = train_list
    val_list = val_list
    test_list = test_list

    train_image_depth_pairs = load_image_depth_pairs(os.path.join(root, train_list))
    val_image_depth_pairs = load_image_depth_pairs(os.path.join(root, val_list))
    test_image_depth_pairs = load_image_depth_pairs(os.path.join(root, test_list))

    train_pairs = train_image_depth_pairs[:train_size] if train_size else train_image_depth_pairs
    val_pairs = val_image_depth_pairs[:val_size] if val_size else val_image_depth_pairs
    test_pairs = test_image_depth_pairs

    # Dataset and Dataloader
    g = torch.Generator()
    g.manual_seed(0)

    train_batch_size, val_batch_size, test_batch_size = batch_size, batch_size, batch_size

    train_dataset = ImageDepthDataset(train_image_folder, train_depth_folder, transform, train_pairs, augmentation)
    sampler = RandomSampler(train_dataset, generator=g)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=sampler)

    val_dataset = ImageDepthDataset(train_image_folder, train_depth_folder, transform, val_pairs, augmentation)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    test_dataset = TestImageDepthDataset(test_image_folder, test_depth_folder, transform, test_pairs, augmentation)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_dataloader(image_size, mode, set_size, batch_size,  train_list="train_list.txt", val_list="val_list.txt", test_list="test_list.txt", root="/cluster/courses/cil/monocular_depth/data", augmentation='none'):
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform # preprocessing (normalization etc.)

    image_folder = os.path.join(root, "train") if mode in ['train', 'val'] else os.path.join(root, "test")
    depth_folder = image_folder if mode in ['train', 'val'] else os.path.join(root, "predictions")

    list_ = {'train': train_list, 'val': val_list, 'test': test_list}[mode]
    image_depth_pairs = load_image_depth_pairs(os.path.join(root, list_))

    pairs = image_depth_pairs[:set_size] if set_size and mode in ['train', 'val'] else image_depth_pairs

    dataset = None
    if mode in ['train', 'val']:
        dataset = ImageDepthDataset(image_folder, depth_folder, transform, pairs, augmentation)
    else:
        dataset = TestImageDepthDataset(image_folder, depth_folder, transform, pairs, augmentation)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
