import torch
from MiDaS.midas.dpt_depth import DPTDepthModel

old_midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

midas = DPTDepthModel(
    path="checkpoints/my_midas_model.pt",
    backbone="vitl16_384",
    non_negative=True,
)
midas.eval().to(device)