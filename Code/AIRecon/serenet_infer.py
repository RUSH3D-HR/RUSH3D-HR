import torch
import AIRecon
import numpy as np
import tifffile as tiff
import torch.nn.functional as F
from utils import*

model_path = './reconstruction/source/recon_model/RUSH3D_20240530_ideal_dz1.pth'
model_weight = torch.load(model_path)['model']
model = AIRecon.make(model_weight, load_sd=True).cuda()
model.eval()

img = torch.from_numpy(tiff.imread('C:/Data/Test7_20X_C2/Reconstruction/realign/Test7_20X_S1_C2_B16_T1_realign_merge.tif').astype(np.float32)).cuda()
psf = get_psf_shift_pt("D:/LWJ/Code/recon_rush3d/reconstruction/source/psf/16",depth=73).cuda()

res = model(img,psf)
tiff.imwrite('C:/Data/Test7_20X_C2/test.tif', res.cpu().numpy())