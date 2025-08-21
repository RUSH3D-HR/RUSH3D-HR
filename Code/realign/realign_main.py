import time
import json
import torch
import tifffile
import torch.nn.functional as F
import numpy as np


def make_coord(shape, ranges=None, flatten=True):
    '''
    Make coordinates at grid centers.
    '''
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    # print('ret',ret.shape)
    return ret


def grid_sample_for_lf_gpu(src_lf, center_pt, scale, startX, startY):
    '''
    intput:    
        src_lf: the input raw lf stack-->torch.tensor, (90,2048,2048)
        center_pt: the center point of the raw lf-->list
        scale: grid sample scale
    return: 
        winger-->torch.tensor, (90,2025,2025)
    '''
    src_h, src_w = src_lf.shape[-2], src_lf.shape[-1]
    cv_pt = [1023, 1023]
    dst_h, dst_w = 2025, 2025

    lf = torch.zeros((src_lf.shape[0],2048,2048),dtype=torch.float32)
    lf[:,startY:startY+src_h,startX:startX+src_w] = src_lf
    # tifffile.imwrite("F:/SLiMWare3.1/test/data_f1p4/ROI_60X_C2/realign/LF_padding.tif", lf.to(torch.float16).cpu().numpy().astype(np.uint16))


    if center_pt is None:
        center_pt = cv_pt

    coord_test = make_coord((2048, 2048), flatten=False).flip(-1).unsqueeze(0)  # 1,2048,2048,2

    #显微系统设置中心点
    center_system = coord_test[:, cv_pt[0]:cv_pt[0] + 1, cv_pt[1]:cv_pt[1] + 1, :]  # 1,1,1,2
    center_system = center_system + torch.tensor([center_pt[0] - cv_pt[0], center_pt[1] - cv_pt[1]],dtype=torch.float32).flip(-1).reshape(1,1,1,2) * 2 / 2048
    #中心点定死
    coord_center = coord_test - coord_test[:, cv_pt[0], cv_pt[1], :]
    #生成以定死中心点为中心的，2025*2025的网格用于采样
    new_coord = coord_center[:, cv_pt[0] - dst_h // 2:cv_pt[0] + dst_h // 2 + 1, cv_pt[1] - dst_w // 2:cv_pt[1] + dst_w // 2 + 1,:]  # 1,2025,2025,2
    lf = lf.unsqueeze(axis=0) # 1,90,2048,2048
    query_coord = new_coord * scale + center_system # 1,2025,2025,2
    lf = F.grid_sample(lf.cuda(), query_coord.cuda(), mode='bilinear', align_corners=False).squeeze()  # 1,90,2025,2025

    return lf


def White_Balance_Debleaching(LF_wdf:torch.tensor,white_wdf:torch.tensor, nx, ny):
    '''
    LF_wdf: Wigner of LF after resize and realign, tensor(225,135,135)
    White_wdf: Wigner of White image after realign,tensor (225,135,135)
    '''
    crop = (white_wdf.shape[-1] - LF_wdf.shape[-1]) // 2
    white_wdf = white_wdf[:, crop:crop+LF_wdf.shape[-1], crop:crop+LF_wdf.shape[-1]]
    base = torch.mean(white_wdf, axis=(1,2), keepdims=True) # (225,1,1)
    white = white_wdf/base
    LF_wdf = LF_wdf/white # 90,225,135,135

    baseNorm = torch.sum(LF_wdf[0].unsqueeze(0),dim=[2,3],keepdims=True) # 1,225,1,1
    LF_wdf = LF_wdf / torch.sum(LF_wdf,dim=[2,3],keepdims=True)*baseNorm # 90,225,1,1

    loc = (LF_wdf[0,113,:,:] != 0).nonzero()[0] #()
    LF_wdf = LF_wdf[:,:,loc[0]:loc[0]+ny,loc[1]:loc[1]+nx]

    return LF_wdf

def realign_reshape(input_img, white_img, center_pt, scale, nx, ny, startX, startY):
    """
    input
        input_img: shape(90,2048,2048), dtype=torch.tensor
        white_img: shape(225,135,135), dtype=torch.tensor
        center_pt: list[float,float]
        scale: float
    return 
        merged_wigner: shape(225,405,405), dtype=torch.tensor.float32
    """
    ## preprocess
    if input_img.shape[-1] == 2048 and nx == 135 and ny == 135: # 全画幅
        startX,startY = 0,0
    elif input_img.shape[-1] == 2048 and (nx != 135 or ny != 135): # 重建ROI
        input_img = input_img[:,startY:startY+ny*15,startX:startX+nx*15]
    else: # 采集ROI
        pass

    ## imresize and realign
    # start = time.time()
    resized_LF = grid_sample_for_lf_gpu(input_img, center_pt, scale, startX, startY) # 90,2025,2025
    # tifffile.imwrite("F:/SLiMWare3.1/test/data_f1p4/ROI_60X_C2/realign/LF_cut.tif", resized_LF.to(torch.float16).cpu().numpy().astype(np.uint16))

    wigner = resized_LF.reshape(input_img.shape[0], 135, 15, 135, 15).permute(0, 2, 4, 1, 3).reshape(input_img.shape[0], 225, 135, 135)
    # print('imresize and realign time:',end-start)
    # tifffile.imwrite("F:/SLiMWare3.1/test/data_f1p4/ROI_60X_C2/realign/wigner.tif", wigner[0].to(torch.float16).cpu().numpy().astype(np.uint16))


    ## White balance
    wigner_white = White_Balance_Debleaching(wigner, white_img.cuda(), nx, ny) # 90,225,135,135

    ## Merge wigner
    # merged_wigner = wigner_white.reshape(3, 3, 225, 135, 135).permute(2, 3, 0, 4, 1).reshape(225, 405, 405)
    # return merged_wigner

    return wigner_white


def realign_merge(wigner_white, order, group_mode, start_frame, n_shift=3): #order 3*3 np.array
    n, v, h, w = wigner_white.shape
    order = order.flatten().tolist()
    if group_mode == 1:  ## n,225,135,135->n-8,225,405,405
        merged_wigner = torch.zeros(n - 8, v, h*n_shift, w*n_shift)
        for i in range(merged_wigner.shape[0]):
            wigner_tmp = wigner_white[i:i + n_shift**2]
            wigner_tmp = torch.roll(wigner_tmp, (i+start_frame) % n_shift**2, 0)
            merged_wigner[i] = wigner_tmp[order].reshape(n_shift, n_shift, v, h, w).permute(2, 3, 0, 4, 1).reshape(v, h*n_shift, w*n_shift)
    else:  ## n,225,135,135-> n//9,225,405,405
        merged_wigner = torch.zeros(n//n_shift**2, v, h*n_shift, w*n_shift)
        for i in range(merged_wigner.shape[0]):
            wigner_tmp = wigner_white[i * n_shift**2:(i + 1) * n_shift**2]
            merged_wigner[i] = wigner_tmp[order].reshape(n_shift, n_shift, v, h, w).permute(2, 3, 0, 4, 1).reshape(v, h*n_shift, w*n_shift)

    return merged_wigner


if __name__ == "__main__":
    ## parameters
    lf_raw = tifffile.imread('Y:/3_Personal/yanjun/data/test31_20X_C2_C3/capture/test31_20X_C2_0.tiff')
    
    input_img = lf_raw[:9]
    scan_config = torch.tensor([0, 7, 6, 1, 8, 5, 2, 3, 4])
    white_img = tifffile.imread('../source/realign/white_img.tif')
    # agent = json.load(open('Y:/3_Personal/yanjun/data/test31_20X_C2_C3/agent.json'))
    center_pt = [1022,1018.5]
    scale = 1
    nx, ny = 101, 101
    startX, startY = 202, 220
    start_frame = 0
    VSR = 0
    group_mode = 1

    ## resize and realign
    input_img = torch.tensor(input_img,dtype=torch.float32).cuda()
    white_img = torch.tensor(white_img.astype(np.float32),dtype=torch.float32).cuda()
    start = time.time()
    wigner_white = realign_reshape(input_img, white_img, center_pt, scale, nx, ny, startX, startY) # 90,225,135,135
    print('realign_reshape time:',time.time()-start)

    ## merge wigners
    if VSR == 1:
        print('Demotion')
    else:
        batch_lf_stack = realign_merge(wigner_white, scan_config, group_mode, start_frame, n_shift=3)
        print('merge wigners time:',time.time()-start)

    tifffile.imwrite('Y:/3_Personal/yanjun/data/test31_20X_C2_C3/realign/realign.tif',batch_lf_stack.cpu().numpy()) 