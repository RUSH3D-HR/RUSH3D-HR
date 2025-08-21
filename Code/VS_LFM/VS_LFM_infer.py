"""Testing script for LF-image SR.
Train a model:
        python VS_LFM_infer.py
See Readme.md for more testing details.
"""
import torch
import torchvision.transforms as transforms
# import os
# import cv2
# import numpy as np
from utils import log_out, time
from torchvision.transforms.functional import gaussian_blur
import torch.nn.functional as F

def detect_artefact(merged_wigner, device):
    kernel = torch.zeros((9, 9)).to(device)
    kernel[1::3, 1::3] = 1
    kernel[kernel != 1] = -9 / (9*9-9)

    sim_cv = merged_wigner[0, 0]
    sim_cv_pad = F.pad(sim_cv.unsqueeze(0).unsqueeze(0), [4,4,4,4], mode='reflect')
    pattern = F.conv2d(sim_cv_pad, kernel.unsqueeze(0).unsqueeze(0), padding=0).squeeze() # 3h 3w
    pattern -= pattern.mean()
    pattern = torch.abs(pattern)
    pattern -= pattern.min()
    pattern /= pattern.max()

    return pattern.mean()


def VS_LFM_inference(model, merged_wigner, scan_order, start_frame, device):
    '''
    merged_wigner: Tensor[CPU] t,81,3h,3w

    '''
    ## motion artifacts dectection kernel
    t = time()
    kernel = torch.zeros((9, 9)).to(device)
    kernel[1::3, 1::3] = 1
    kernel[kernel != 1] = -9 / (9*9-9)

    N, A, H, W = merged_wigner.shape[0], merged_wigner.shape[1], merged_wigner.shape[2], merged_wigner.shape[3]
    output = torch.zeros((N, A, H, W), dtype=torch.float32)
    view_mask = torch.zeros((N, A, H, W), dtype=torch.float32)
    data_LR = merged_wigner[:,:,1::3,1::3] # t 81 h w
    log_out(f'VS preprocess takes {(time() - t):1f}')
    for i in range(N):
        with torch.no_grad():
            t = time()
            data_SR = model(data_LR[i:i + 1, :, :, :]).squeeze() # 81 3h 3w
            data_SR = torch.clamp(data_SR, data_LR.min().item(), data_LR.max().item())  # Limit the value to the range of ori LR
            log_out(f"VS model takes {(time() - t):1f}")
            cur_frame = merged_wigner[i].to(device) # 81 3h 3w
            t = time()
            for v in range(cur_frame.shape[0]):
                sim_cv = cur_frame[v] # 3h 3w
                vs_cv = data_SR[v] # 3h 3w
                sim_cv_pad = F.pad(sim_cv.unsqueeze(0).unsqueeze(0), [4,4,4,4], mode='reflect')
                pattern = F.conv2d(sim_cv_pad, kernel.unsqueeze(0).unsqueeze(0), padding=0).squeeze() # 3h 3w
                pattern -= pattern.mean()
                pattern = torch.abs(pattern)
                pattern -= pattern.min()
                pattern /= pattern.max()
                mask = pattern.clone()
                mask[mask < 0.1] = 0
                mask[(mask > 0) & (mask < 1)] += 0.4
                mask[mask > 1] = 1
                # mask = gaussian_blur(mask.unsqueeze(0).unsqueeze(0), [7, 7])
                # mask[mask > 0.3] = 1
                # mask[mask <= 0.3] = 0
                # mask = gaussian_blur(mask, [7, 7]).squeeze()
                fix_cv = (1 - mask) * sim_cv + mask * vs_cv
                fix_cv = torch.clip(fix_cv, 0, 65535)
                output[i, v, :, :] = fix_cv.cpu()
                view_mask[i, v, :, :] = mask
            log_out(f"VS mask takes {(time() - t):1f}")
    return output, view_mask

def VS_LFM_inference_trt(model, data_LR, scan_order, wigner_idx):
    # h, w, memc = scan_memc(3, scan_order,scale=3)
    with torch.no_grad():
        data_SR = model(data_LR)
    data_SR = torch.clamp(data_SR, data_LR.min(), data_LR.max())  # Limit the value to the range of ori LR
    # output= transforms.functional.affine(data_SR, 
    #                                     translate = memc[wigner_idx%9].tolist(),
    #                                     angle=0,scale=1,shear=0)
    
    return data_SR