import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader

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
        return image, depth_file_name, predictions, uncertainty