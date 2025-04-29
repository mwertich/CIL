from model import MiDaSUQ
from utils.dataloader import get_dataloaders
from utils.utils import torch_seed
from utils.loss_funcs import scale_invariant_rmse
from datetime import datetime
import torch
import numpy as np


def predict_model(model, test_loader, image_size, device):
    model.eval()
    with torch.no_grad():
        for images, out_paths in test_loader:
            images = images.to(device)

            preds, _ = model(images)
            preds_resized = torch.nn.functional.interpolate(
                preds.unsqueeze(1), size=image_size, mode="bicubic", align_corners=False
            )
            eps = 1e-8
            preds_resized = preds_resized.clamp(min=eps) # for numerical stability for RMSE to avoid nan values due to log(0)
            for pred, out_path in zip(preds_resized, out_paths):
                np.save(out_path, pred.cpu())


if __name__ == "__main__":
    torch_seed()
    
    run_id = datetime.now().strftime("%y%m%d_%H%M%S")
    print('---------------- Run id:', run_id, '----------------')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = [426, 560]
    _, _, test_loader = get_dataloaders(image_size=image_size)

    model = MiDaSUQ(backbone="vitl16_384")
    state_dict = torch.load("models/best_model_epoch_2.5.pth", map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print("✅ Loaded fine-tuned MiDaS model.")
    
    predict_model(model, test_loader, image_size, device)