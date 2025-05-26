# reproducibility
from utils.utils import torch_seed
torch_seed(seed=0)

import os
import sys
import argparse
import torch
import torch.nn as nn
from utils.dataloader import get_dataloaders
from utils.visualization import visualize_prediction_with_ground_truth, visualize_prediction_without_ground_truth
from utils.loss_funcs import DepthUncertaintyLoss, scale_invariant_rmse, SobelEdgeLoss
from utils.utils import torch_seed
from model import MiDaSUQ
from evaluate_notebook import evaluate_model_notebook
from predict import predict_model
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


# Main training function
def finetune_model(model, train_loader, val_loader, out_path, epochs=5, lr=1e-5, save_every_epoch=True, grad_consistency_loss_lambda=0.0):

    model.to(device)
    model.train()  # set to train mode

    # Define loss and optimizer
    # criterion = nn.L1Loss()  # or nn.MSELoss()
    criterion = DepthUncertaintyLoss()
    gradloss_func = SobelEdgeLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # evaluate initial model
    evaluate_model_notebook(model, val_loader, device, uq=True)

    # Training loop
    for epoch in range(1, epochs + 1):
        batch_losses = []
        running_loss = 0.0
        time = datetime.now()
        print(f"\n🕒 [{time.strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch}/{epochs}")
        for images, depths in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            images, depths = images.to(device), depths.to(device)

            optimizer.zero_grad()
            pred_depths, pred_vars = model(images)

            pred_depths_resized = torch.nn.functional.interpolate(
                pred_depths.unsqueeze(1),
                size=depths.shape[-2:],
                mode="bicubic",
                align_corners=False
            ).squeeze(1)

            pred_vars_resized = torch.nn.functional.interpolate(
                pred_vars.unsqueeze(1),
                size=depths.shape[-2:],
                mode="bicubic",
                align_corners=False
            ).squeeze(1)

            gradloss = gradloss_func(pred_depths_resized.unsqueeze(1), depths)
            loss = criterion(pred_depths_resized, pred_vars_resized, depths.squeeze(1))
            loss += grad_consistency_loss_lambda * gradloss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_losses.append(loss.item())

        print(f"✅ Epoch [{epoch}/{epochs}] finished. Loss: {running_loss/len(train_loader):.4f}")
        evaluate_model_notebook(model, val_loader, device, uq=True, epoch=epoch)

        # Save batch losses after each epoch
        losses_path = f'models/batch_losses_{run_id}_{epoch}.npy'
        np.save(losses_path, np.array(batch_losses))
        print(f"💾 Batch losses saved to {losses_path}")
        if save_every_epoch:
            # Save model after each epoch
            model_path = f'models/model_{run_id}_finetuned_{epoch}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"💾 Model saved to {model_path}")

    # Save fine-tuned model
    torch.save(model.state_dict(), out_path)
    print(f"✅ Fine-tuned model saved to {out_path}")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-p', '--pretrained', default='../.cache/torch/hub/checkpoints/dpt_large_384.pt', type=str,
                      help='pretrained model path (default: ../.cache/torch/hub/checkpoints/dpt_large_384.pt)')
    args.add_argument('-trainl', '--train_list', default="train_list.txt", type=str, 
                      help='Path to train list (default: train_list.txt)')
    args.add_argument('-vall', '--val_list', default="val_list.txt", type=str, 
                      help='Path to val list (default: val_list.txt)')
    args.add_argument('-testl', '--test_list', default="test_list.txt", type=str, 
                      help='Path to test list (default: test_list.txt)')
    args.add_argument('-e', '--epochs', default=1, type=int,
                      help='number of epochs for finetuning (default: 1)')
    args.add_argument('-t', '--train_size', default=0, type=int,
                      help='training set size (default: 0 = all)')
    args.add_argument('-v', '--val_size', default=0, type=int,
                      help='validation set size (default: 0 = all)')
    args.add_argument('-b', '--batch_size', default=1, type=int,
                      help='batch size for dataloaders (default: 1)')
    args.add_argument('-l', '--learning_rate', default=1e-5, type=float,
                      help='batch size for dataloaders (default: 1e-5)')
    args.add_argument('-a', '--augmentation', default = 'none', type = str,
                      help='should the data be augmented? Possible values: sharpen, smooth, none (default: none)')
    args.add_argument('-f', '--no_filter_head', action='store_true', help='Determine whether the last head in MiDaS should be filtered')
    args.add_argument('-save', '--save_every_epoch', action='store_true', help='Save after each epoch')
    args.add_argument("--cluster-root", type=str, default="/cluster/courses/cil/monocular_depth/data/")
    args.add_argument('-gl', default=0.0, type=float, 
                      help="The weight on an additional depth map gradient consistency term added to the loss fucntion")
    config = args.parse_args()

    run_id = datetime.now().strftime("%y%m%d_%H%M%S")
    print('---------------- Run id:', run_id, '----------------')
    print(' '.join(sys.argv))
    print(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and transforms
    # model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    model = MiDaSUQ(backbone="vitl16_384")
    state_dict = torch.load(config.pretrained, map_location=device)
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if "scratch.output_conv.4." not in k  # Exclude final conv layer
    }
    model.load_state_dict(state_dict if config.no_filter_head else filtered_state_dict, strict=False)

    image_size = [426, 560]
    train_loader, val_loader, test_loader = get_dataloaders(image_size, config.train_size, config.val_size, config.batch_size, train_list=config.train_list, val_list=config.val_list, test_list=config.test_list, augmentation=config.augmentation)

    # train_loader = get_dataloader(image_size=image_size, mode='train', set_size=config.train_size, batch_size=config.batch_size)
    # val_loader   = get_dataloader(image_size=image_size, mode='val', set_size=config.val_size, batch_size=config.batch_size)
    # test_loader  = get_dataloader(image_size=image_size, mode='test', set_size=None, batch_size=config.batch_size)

    finetune_model(model, train_loader, val_loader, out_path=f"models/model_{run_id}_finetuned.pth", epochs=config.epochs, lr=config.learning_rate, save_every_epoch=config.save_every_epoch, grad_consistency_loss_lambda=config.gl)

    # Reload the architecture
    # model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    # model.to(device)

    # Load the fine-tuned weights
    # model.load_state_dict(torch.load("models/model_finetuned_final.pth", map_location=device))
    # model = MiDaSUQ(backbone="vitl16_384")
    # state_dict = torch.load("models/model_finetuned_trained.pth", map_location=device)
    # model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    #print("✅ Loaded fine-tuned MiDaS model.")
    #evaluate_model_notebook(model, val_loader, device, uq=True, epoch=10)
    #visualize_prediction_with_ground_truth(model, val_loader, run_id, image_size, device, num_images=10)
    # predict_model(model, test_loader, image_size, device)
    #visualize_prediction_without_ground_truth(model, test_loader, run_id,  image_size, device, num_images=10)