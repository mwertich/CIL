from model import MiDaSUQ
from utils.dataloader import get_dataloader
from utils.visualization import visualize_prediction_with_ground_truth, visualize_prediction_without_ground_truth
from utils.utils import torch_seed
from datetime import datetime
import argparse
import torch


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-p', '--pretrained', default=None, type=str,
                      help='pretrained model path (default: None)')
    args.add_argument('-v', '--val_size', default=5, type=int,
                      help='validation set size (default: 5)')
    args.add_argument('-b', '--batch_size', default=1, type=int,
                      help='batch size for dataloaders (default: 1)')
    config = args.parse_args()
    
    torch_seed()
    
    run_id = datetime.now().strftime("%y%m%d_%H%M%S")
    print('---------------- Run id:', run_id, '----------------')
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = [426, 560]
    val_loader  = get_dataloader(image_size=image_size, mode='val', set_size=config.val_size, batch_size=config.batch_size)
    test_loader = get_dataloader(image_size=image_size, mode='test', set_size=None, batch_size=config.batch_size)

    model = MiDaSUQ(backbone="vitl16_384")
    state_dict = torch.load(config.pretrained, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print("✅ Loaded fine-tuned MiDaS model.")
    
    visualize_prediction_with_ground_truth(model, val_loader, run_id, image_size, device, num_images=10, map_error=True)
    visualize_prediction_without_ground_truth(model, test_loader, run_id,  image_size, device, num_images=10)