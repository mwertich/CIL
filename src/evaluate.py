from model import MiDaSUQ
from utils.dataloader import get_dataloader
from utils.utils import torch_seed
from utils.loss_funcs import scale_invariant_rmse
from datetime import datetime
import argparse
import torch


def evaluate_model(model, val_loader, epoch, device):
    model.eval()
    total_rmse = 0.0
    with torch.no_grad():
        for images, depths in val_loader:
            images = images.to(device)
            depths = depths.to(device)
            
            depth, _ = model(images)
            depth_resized = torch.nn.functional.interpolate(
                depth.unsqueeze(1), size=depths.shape[-2:], mode="bicubic", align_corners=False
            )

            eps = 1e-8
            preds_resized = depth_resized.clamp(min=eps) # for numerical stability for RMSE to avoid nan values due to log(0)

            loss = scale_invariant_rmse(preds_resized, depths)
            total_rmse += loss.item()

    avg_rmse = total_rmse / len(val_loader)
    if epoch is not None:
        print(f"✅ Scale-Invariant RMSE after epoch {epoch}: {avg_rmse:.4f}")
    else:
        print(f"✅ Scale-Invariant RMSE: {avg_rmse:.4f}")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-p', '--pretrained', default=None, type=str,
                      help='pretrained model path (default: None)')
    args.add_argument('-v', '--val_size', default=None, type=int,
                      help='validation set size (default: None)')
    args.add_argument('-b', '--batch_size', default=None, type=int,
                      help='batch size for dataloaders (default: None)')
    config = args.parse_args()

    torch_seed()
    
    run_id = datetime.now().strftime("%y%m%d_%H%M%S")
    print('---------------- Run id:', run_id, '----------------')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = [426, 560]
    val_loader = get_dataloader(image_size=image_size, mode='val', set_size=config.val_size, batch_size=config.batch_size)

    model = MiDaSUQ(backbone="vitl16_384")
    state_dict = torch.load(config.pretrained, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print("✅ Loaded fine-tuned MiDaS model.")
    
    evaluate_model(model, val_loader, None, device)
    