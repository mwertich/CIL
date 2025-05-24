from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch



def evaluate_model_notebook(model, val_loader, device="cuda", uq=False, epoch=0):
    """Evaluate the model and compute metrics on validation set"""
    model.eval()
    
    mae = 0.0
    rmse = 0.0
    rel = 0.0
    delta1 = 0.0
    delta2 = 0.0
    delta3 = 0.0
    sirmse = 0.0
    
    total_samples = 0
    target_shape = None
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            if target_shape is None:
                target_shape = targets.shape
            

            # Forward pass
            if uq:
                outputs, _ = model(inputs)
            else:
                outputs = model(inputs)
            
            # Resize outputs to match target dimensions
            outputs = nn.functional.interpolate(
                outputs.unsqueeze(1),
                size=targets.shape[-2:],  # Match height and width of targets
                mode='bilinear',
                align_corners=True
            )
            
            # Calculate metrics
            abs_diff = torch.abs(outputs - targets)
            mae += torch.sum(abs_diff).item()
            rmse += torch.sum(torch.pow(abs_diff, 2)).item()
            rel += torch.sum(abs_diff / (targets + 1e-6)).item()
            
            # Calculate scale-invariant RMSE for each image in the batch
            for i in range(batch_size):
                # Convert tensors to numpy arrays
                pred_np = outputs[i].cpu().squeeze().numpy()
                target_np = targets[i].cpu().squeeze().numpy()
                
                EPSILON = 1e-6
                
                valid_target = target_np > EPSILON
                if not np.any(valid_target):
                    continue
                
                target_valid = target_np[valid_target]
                pred_valid = pred_np[valid_target]
                
                log_target = np.log(target_valid)
                
                pred_valid = np.where(pred_valid > EPSILON, pred_valid, EPSILON)
                log_pred = np.log(pred_valid)
                
                # Calculate scale-invariant error
                diff = log_pred - log_target
                diff_mean = np.mean(diff)
                
                # Calculate RMSE for this image
                sirmse += np.sqrt(np.mean((diff - diff_mean) ** 2))
            
            # Calculate thresholded accuracy
            max_ratio = torch.max(outputs / (targets + 1e-6), targets / (outputs + 1e-6))
            delta1 += torch.sum(max_ratio < 1.25).item()
            delta2 += torch.sum(max_ratio < 1.25**2).item()
            delta3 += torch.sum(max_ratio < 1.25**3).item()
            
            # Free up memory
            del inputs, targets, outputs, abs_diff, max_ratio
            
        # Clear CUDA cache
        torch.cuda.empty_cache()
    
    # Calculate final metrics using stored target shape
    total_pixels = target_shape[1] * target_shape[2] * target_shape[3]  # channels * height * width
    mae /= total_samples * total_pixels
    rmse = np.sqrt(rmse / (total_samples * total_pixels))
    rel /= total_samples * total_pixels
    sirmse = sirmse / total_samples
    delta1 /= total_samples * total_pixels
    delta2 /= total_samples * total_pixels
    delta3 /= total_samples * total_pixels
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'siRMSE': sirmse,
        'REL': rel,
        'Delta1': delta1,
        'Delta2': delta2,
        'Delta3': delta3
    }
    
    print(f"Scores after epoch {epoch}: " + str({k: f"{float(v):.4f}"[-6:] for k, v in metrics.items()}))

    return metrics


def evaluate_uncertainty(model, val_loader, device="cuda", epoch=0):
    """Evaluate the model capability to quatify the uncertainty of its predictions"""
    model.eval()
   
    total_samples = 0
    target_shape = None
    total_accurate_certain = 0
    total_certain = 0
    total_inaccurate_uncertain = 0
    total_inaccurate = 0
    total_pavpu_pixels = 0

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            pixel_num = inputs.size(2) * inputs.size(3)
            total_samples += batch_size
            
            if target_shape is None:
                target_shape = targets.shape
            

            # Forward pass
            outputs, variance = model(inputs)
            
            # Resize outputs to match target dimensions
            outputs = nn.functional.interpolate(
                outputs.unsqueeze(1),
                size=targets.shape[-2:],  # Match height and width of targets
                mode='bilinear',
                align_corners=True
            )
            
            variance = nn.functional.interpolate(
                variance.unsqueeze(1),
                size=targets.shape[-2:],  # Match height and width of targets
                mode='bilinear',
                align_corners=True
            )
    
            median_uncertainty = torch.median(variance.squeeze(1).view(batch_size, -1), dim=1).values.view(-1, 1, 1, 1)
            max_ratio = torch.max(outputs / (targets + 1e-6), targets / (outputs + 1e-6))
            certain = torch.sum((variance < median_uncertainty)).item()
            accurate_and_certain = torch.sum((max_ratio < 1.25) * (variance < median_uncertainty)).item()
            assert accurate_and_certain <= certain, "Inconsistent counts: accurate_and_certain should be less than or equal to accurate"
            inaccurate = torch.sum(max_ratio >= 1.25).item() 
            inaccurate_and_uncertain = torch.sum((max_ratio >= 1.25) * (variance >= median_uncertainty)).item()
            assert inaccurate_and_uncertain <= inaccurate, "Inconsistent counts: inaccurate_and_uncertain should be less than or equal to uncertain"
            total_accurate_certain += accurate_and_certain
            total_certain += certain
            total_inaccurate_uncertain += inaccurate_and_uncertain
            total_inaccurate += inaccurate
            total_pavpu_pixels += (accurate_and_certain + inaccurate_and_uncertain)

    
    # Calculate final metrics using stored target shape
    total_pixels = target_shape[1] * target_shape[2] * target_shape[3]  # channels * height * width
    pa = total_accurate_certain / total_certain if total_certain > 0 else 0
    pu = total_inaccurate_uncertain / total_inaccurate if total_inaccurate > 0 else 0
    pavpu = total_pavpu_pixels / (total_samples * total_pixels)
        
    metrics = {
        'PA': pa,
        'PU': pu,
        'PAvPU': pavpu,
    }
    
    print(f"Uncertainty metrics after epoch {epoch}: " + str({k: f"{float(v):.4f}" for k, v in metrics.items()}))

    return metrics



if __name__ == "__main__":
    import argparse
    from utils.dataloader import get_dataloader
    from model import MiDaSUQ

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-p', '--pretrained', required=True, type=str, help='pretrained model path')
    args.add_argument('--val_list', default="val_list.txt", type=str, help='Path to val list (default: val_list.txt)')
    args.add_argument('--val_size', default=0, type=int, help='validation set size (default: 0 = all)')
    args.add_argument('-b', '--batch_size', default=1, type=int, help='batch size for dataloaders (default: 1)')
    config = args.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiDaSUQ(backbone="vitl16_384")
    state_dict = torch.load(config.pretrained, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    image_size = [426, 560]
    
    val_loader = get_dataloader(image_size, "val",
                                config.val_size, config.batch_size,
                                val_list=config.val_list, 
                                root="src/data")

    evaluate_uncertainty(model, val_loader, device=device, epoch=0)
    evaluate_model_notebook(model, val_loader, device=device, uq=True, epoch=0)

