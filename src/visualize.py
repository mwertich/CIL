from model import MiDaSUQ
from utils.dataloader import get_dataloaders
from utils.visualization import visualize_prediction_with_ground_truth, visualize_prediction_without_ground_truth
from utils.utils import torch_seed
from datetime import datetime
import torch


if __name__ == "__main__":
    
    torch_seed()
    
    run_id = datetime.now().strftime("%y%m%d_%H%M%S")
    print('---------------- Run id:', run_id, '----------------')
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = [426, 560]
    _, val_loader, test_loader = get_dataloaders(image_size=(426, 560))

    model = MiDaSUQ(backbone="vitl16_384")
    state_dict = torch.load("models/best_model_epoch_2.5.pth", map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print("✅ Loaded fine-tuned MiDaS model.")
    
    visualize_prediction_with_ground_truth(model, val_loader, run_id, image_size, device, num_images=10)
    visualize_prediction_without_ground_truth(model, test_loader, run_id,  image_size, device, num_images=10)