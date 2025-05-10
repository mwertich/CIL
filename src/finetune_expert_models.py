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
from datetime import datetime
from utils.loss_funcs import scale_invariant_rmse
from utils.dataloader import get_dataloader
from model import MiDaSUQ
from utils.visualization import denormalize_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = [426, 560]


def evaluate_model(model, val_loader, epoch=0):
    model.eval()
    total_sirmse = 0.0
    with torch.no_grad():
        for images, depths in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            depths = depths.to(device)

            preds = model(images)
            if config.uq:
                preds = preds[0]
            preds_resized = torch.nn.functional.interpolate(
                preds.unsqueeze(1), size=depths.shape[-2:], mode="bicubic", align_corners=False
            )

            eps = 1e-8
            preds_resized = preds_resized.clamp(min=eps) # for numerical stability for RMSE to avoid nan values due to log(0)

            loss = scale_invariant_rmse(preds_resized, depths)
            total_sirmse += loss.item()

    sirmse = total_sirmse / len(val_loader)
    print(f"✅ Scale-Invariant RMSE after epoch {epoch}: {sirmse:.4f}")



# Main training function
def finetune_model(model, train_loader, val_loader, out_path, epochs=5, lr=1e-5, save_model = True):

    model.to(device)
    model.train()  # set to train mode

    # Define loss and optimizer
    criterion = nn.L1Loss()  # or nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #evaluate initial model
    #evaluate_model(model, val_loader, 0)

    # Training loop
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
         # Log starting time
        time = datetime.now()
        print(f"\n🕒 [{time.strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch}/{epochs}")

        for images, depths in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            images, depths = images.to(device), depths.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            outputs_resized = torch.nn.functional.interpolate(
                outputs.unsqueeze(1),
                size=depths.shape[-2:],
                mode="bicubic",
                align_corners=False
            ).squeeze(1)

            loss = criterion(outputs_resized, depths.squeeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"✅ Epoch [{epoch}/{epochs}] finished. Loss: {running_loss/len(train_loader):.4f}")
        evaluate_model(model, val_loader, epoch)
        # Save model after each epoch
        
        model_path = f"models/model_finetuned_epoch_{epoch}.pth"
        #torch.save(model.state_dict(), model_path)
        #print(f"💾 Model saved to models/{model_path}")

    # Save fine-tuned model
    if save_model:
        torch.save(model.state_dict(), out_path)
        print(f"✅ Fine-tuned model saved to {out_path}")


def predict_model(model, test_loader, eps = 1e-8):
    model.eval()
    with torch.no_grad():
        for images, depth_file_names in test_loader:
            images = images.to(device)

            preds = model(images)
            if config.uq:
                preds = preds[0]
            preds_resized = torch.nn.functional.interpolate(
                preds.unsqueeze(1), size=image_size, mode="bicubic", align_corners=False
            )
            preds_resized = preds_resized.clamp(min=eps) # for numerical stability for RMSE to avoid nan values due to log(0)
            for pred, depth_file_name in zip(preds_resized, depth_file_names):
                np.save(depth_file_name, pred.cpu())



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
    plt.savefig(file_name)


def visualize_prediction_with_ground_truth(model, loader, num_images=5):
    model.eval()

    images_shown = 0
    with torch.no_grad():
        for images, depths in loader:
            images = images.to(device)
            depths = depths.to(device)
            pred = model(images)
            if config.uq:
                pred = pred[0]
            pred_resized = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=depths.shape[-2:], mode="bicubic", align_corners=False
            )
            for image, depth, prediction in zip(images, depths, pred_resized): 
                images_shown += 1
                file_name = f"depth_maps/val/depth_map_{images_shown}.png"
                visualize_depth_maps("Depths Map Validation Set", file_name, image, prediction, depth)
                if images_shown >= num_images:
                    return


def visualize_prediction_without_ground_truth(model, test_loader, num_images=5):
    model.eval()
    images_shown = 0
    with torch.no_grad():
        for images, out_paths in test_loader:

            images = images.to(device)
            pred = model(images)
            if config.uq:
                pred = pred[0]
            pred_resized = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=image_size, mode="bicubic", align_corners=False
            )
            for image, prediction, out_path in zip(images, pred_resized, out_paths):
                images_shown += 1
                file_name = f"depth_maps/test/depth_map_{images_shown}.png"
                visualize_depth_maps("Depths Map Test Set", file_name, image, prediction)
                if images_shown >= num_images:
                    return


def main(config):
    root = "src/data"
    #cluster_root = "/cluster/courses/cil/monocular_depth/data/"
    cluster_root = "src/data"
    category = config.category

    if config.category:
        train_list = f"category_lists/{category}_train_list.txt"
        val_list = f"category_lists/{category}_val_list.txt"
        test_list = f"category_lists/{category}_test_list.txt"
    else:
        train_list = f"train_list.txt"
        val_list = f"val_list.txt"
        test_list = f"test_list.txt"

    train_loader = get_dataloader(image_size=image_size, mode='train', set_size=config.train_size, batch_size=config.batch_size, train_list=train_list, val_list=val_list, test_list=test_list, sharpen=False) #19176/23971
    val_loader   = get_dataloader(image_size=image_size, mode='val', set_size=config.val_size, batch_size=config.batch_size, train_list=train_list, val_list=val_list, test_list=test_list, sharpen=False) #4795/23971
    test_loader  = get_dataloader(image_size=image_size, mode='test', set_size=None, batch_size=config.batch_size, train_list=train_list, val_list=val_list, test_list=test_list, sharpen=False) #650/650

    if not config.category and config.pretrained:
        model_path = config.pretrained
        print(f"Load model from {model_path}")
    elif config.category:
        model_path = f"models/model_{category}_finetuned.pth"

    # Reload the architecture
    if config.uq:
        model = MiDaSUQ(backbone="vitl16_384")
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    else:
        model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        if config.pretrained:
            # Load the fine-tuned weight
            model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    

    model.to(device)
    print("✅ Loaded fine-tuned MiDaS model.")

    #finetune_model(model, train_loader, val_loader, out_path=model_path, epochs=config.epochs, save_model=True)

    model.eval()
    print("✅ Evaluate fine-tuned MiDaS model.")
    evaluate_model(model, val_loader)
    evaluate_model_notebook(model, val_loader, device)

    print("✅ Predict with fine-tuned MiDaS model.")
    #visualize_prediction_with_ground_truth(model, val_loader, num_images=10)
    #visualize_prediction_without_ground_truth(model, test_loader, num_images=10)
    #predict_model(model, test_loader)
    print("Finished")




if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Fine-tune MiDaS model for indoor depth estimation.")
    args.add_argument("--pretrained", type=str, default=None, help="Path to model. If left out, a new model is trained")
    args.add_argument("--category", type=str, help="Category (e.g., kitchen, living_room, etc.)")
    args.add_argument("--train-size", type=int, default=None, help="Subset size of training data")
    args.add_argument("--val-size", type=int, default=None, help="Subset size of validaton data")
    args.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    args.add_argument("-b", "--batch-size", type=int, default=4,help="Batch size")
    args.add_argument("--uq", type=bool, default=False)
    config = args.parse_args()

    run_id = datetime.now().strftime("%y%m%d_%H%M%S")
    print('---------------- Run id:', run_id, '----------------')

    main(config)