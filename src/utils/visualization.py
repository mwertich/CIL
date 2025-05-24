import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from pathlib import Path


def denormalize_image(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    return tensor.clamp(0, 1)


def visualize_depth_maps(plt_title, file_name, image, pred_depth, pred_var, ground_truth=None, map_error=False):
    # Convert tensors to numpy for visualization
    img_vis = denormalize_image(image.squeeze(0).cpu())
    img_np = TF.to_pil_image(img_vis)
    pred_depth_np = pred_depth.squeeze().cpu().numpy()
    pred_var_np = pred_var.squeeze().cpu().numpy()
    # Normalize depth maps for display
    pred_depth_disp = (pred_depth_np - pred_depth_np.min()) / (pred_depth_np.max() - pred_depth_np.min())
    pred_var_disp = (pred_var_np - pred_var_np.min()) / (pred_var_np.max() - pred_var_np.min())

    if ground_truth is not None:
        depth_np = ground_truth.squeeze().cpu().numpy()
        gt_disp = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
        captions = ["Input Image", "Ground Truth", "Predicted Depth", "Uncertainty"]
        plot_images = [img_np, gt_disp, pred_depth_disp, pred_var_disp]
        cmaps = [None, "plasma", "plasma", "viridis"]
    else:
        captions = ["Input Image", "Predicted Depth", "Uncertainty"]
        plot_images = [img_np, pred_depth_disp, pred_var_disp]
        cmaps = [None, "plasma", "viridis"]

    if map_error:
        assert ground_truth is not None

        error_np = np.abs(pred_depth_np - depth_np)
        erorr_disp = (error_np - error_np.min()) / (error_np.max() - error_np.min())
        captions.append("Error (absolute value)")
        plot_images.append(erorr_disp)
        cmaps.append('viridis')
    
    # Plot
    ncols = 5 if map_error else 4 if ground_truth is not None else 3

    fig, axs = plt.subplots(1, ncols, figsize=(12, 4))
    fig.suptitle(plt_title)
    for (k, (caption, plot_img)) in enumerate(zip(captions, plot_images)):
        i = k % ncols

        axs[i].set_title(caption)
        axs[i].imshow(plot_img, cmap=cmaps[k])
        axs[i].axis("off")
    
    fig.tight_layout()
    print("Saving visualization map to", file_name, "...")
    fig.savefig(file_name)
    plt.close(fig)


def visualize_prediction_with_ground_truth(model, loader, run_id, image_size, device, num_images=5, map_error=False):
    model.eval()

    out_dir = Path(f'./depth_maps/val/{run_id}')
    os.mkdir(out_dir)

    images_shown = 0
    with torch.no_grad():
        for images, depths in loader:
            images = images.to(device)
            depths = depths.to(device)

            pred_depths, pred_vars = model(images)
            pred_depths_resized = torch.nn.functional.interpolate(
                pred_depths.unsqueeze(1), size=image_size, mode="bicubic", align_corners=False
            )
            
            pred_vars_resized = torch.nn.functional.interpolate(
                pred_vars.unsqueeze(1), size=image_size, mode="bicubic", align_corners=False
            )
            # pred_vars_resized = torch.exp(pred_vars_resized).clamp(min=1e-6)
            for image, depth, pred_depth, pred_var in zip(images, depths, pred_depths_resized, pred_vars_resized): 
                images_shown += 1
                file_name = f"depth_maps/val/{run_id}/midas_uq_depth_map_{images_shown}.png"
                visualize_depth_maps("Depths Map Validation Set", file_name, image, pred_depth, pred_var, depth, map_error)
                # file_name = f"depth_maps/val/midas_uq_var_map_{images_shown}.png"
                # visualize_depth_maps("Uncertainty Map Validation Set", file_name, image, pred_var, uncertainty=True)
                if images_shown >= num_images:
                    return


def visualize_prediction_without_ground_truth(model, test_loader, run_id, image_size, device, num_images=5):
    model.eval()

    out_dir = Path(f'./depth_maps/test/{run_id}')
    os.mkdir(out_dir)

    images_shown = 0
    with torch.no_grad():
        for images, out_paths in test_loader:

            images = images.to(device)

            pred_depths, pred_vars = model(images)
            pred_depths_resized = torch.nn.functional.interpolate(
                pred_depths.unsqueeze(1), size=image_size, mode="bicubic", align_corners=False
            )
            
            pred_vars_resized = torch.nn.functional.interpolate(
                pred_vars.unsqueeze(1), size=image_size, mode="bicubic", align_corners=False
            )
            # pred_vars_resized = torch.exp(pred_vars_resized).clamp(min=1e-6)
            for image, pred_depth, pred_var in zip(images, pred_depths_resized, pred_vars_resized): 
                images_shown += 1
                file_name = f"depth_maps/test/{run_id}/midas_uq_depth_map_{images_shown}.png"
                visualize_depth_maps("Depths Map Test Set", file_name, image, pred_depth, pred_var)
                # file_name = f"depth_maps/val/midas_uq_var_map_{images_shown}.png"
                # visualize_depth_maps("Uncertainty Map Validation Set", file_name, image, pred_var, uncertainty=True)
                if images_shown >= num_images:
                    return
