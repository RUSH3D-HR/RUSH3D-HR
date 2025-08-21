import os
import torch
from .vsnet2 import Net as vs2


def init_vs_lfm(model_path, device):
    """Initialize the model"""
    VS_LFM = vs2(81, 3).to(device)

    VS_LFM = torch.nn.DataParallel(VS_LFM, device_ids=[device])  # Use 1 GPU[0]

    # Load the pretrained model for test
    model = torch.load(model_path)
    VS_LFM.load_state_dict(model)
    VS_LFM.eval()

    return VS_LFM