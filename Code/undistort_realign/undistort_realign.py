# import tifffile
import torch
import numpy as np
import torch.nn.functional as F
# import json
# import time
import os
# import sys
# import cv2
# from skimage.feature import peak_local_max
from scipy import signal
import pickle
# from skimage.filters import rank
# from skimage.morphology import disk, ball
# import torchvision.transforms as transforms
import glob

def sub_undistort_coor(params, center_x, center_y, nx, ny, Nnum=15):
    '''
    计算去畸变子图在原始畸变光场图上的对应坐标
    '''
    H, W = ny*Nnum, nx*Nnum
    gty,gtx = np.mgrid[center_y-(Nnum*ny)//2:center_y+(Nnum*ny)//2+1,center_x-(Nnum*nx)//2:center_x+(Nnum*nx)//2+1]
    gtxy = np.c_[gtx.ravel(), gty.ravel()]
    x_undistorted, y_undistorted = distort_model(params['inv_undistort'],(gtxy[:,0]-14304//2)/100,(gtxy[:,1]-10748//2)/100)
    x_undistorted=x_undistorted*100+14304//2
    y_undistorted=y_undistorted*100+10748//2
    return x_undistorted,y_undistorted

def distort_model(params, x, y):
    '''
    径向畸变模型坐标
    x,y:去畸变后的标准网格坐标
    '''
    fx,fy,cx,cy,k1, k2, k3, p1, p2 = params
    matrix=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    objpoints=np.concatenate((x[:,np.newaxis],y[:,np.newaxis],np.ones_like(y[:,np.newaxis])),axis=1)
    objpoints_rotated=np.matmul(objpoints, matrix)
    objpoints_projected = objpoints_rotated[:, :2] / (objpoints_rotated[:, 2:] + 1e-17)
    shift=objpoints_projected-np.array([cx,cy])

    x_shifted = shift[:,0]
    y_shifted = shift[:,1]
    r2 = x_shifted**2 + y_shifted**2
    x_distorted = x_shifted * (1 + k1*r2 + k2*r2**2 + k3*r2**3) + 2*p1*x_shifted*y_shifted + p2*(r2 + 2*x_shifted**2) + cx
    y_distorted = y_shifted * (1 + k1*r2 + k2*r2**2 + k3*r2**3) + p1*(r2 + 2*y_shifted**2) + 2*p2*x_shifted*y_shifted + cy
    return x_distorted, y_distorted

def determine_scan_order(input_image, order):
    '''
     input:
        center view wigner: (9, ny_2, nx_2), dtype=torch.tensor
        order: (3,3), dtype=np.array
    '''
    var = []
    for i in range(9):
        tmp_order=torch.roll(input_image,i,0)
        realign_reshape=tmp_order[order].reshape(3,3,input_image.shape[1],input_image.shape[2]).permute(2,0,3,1).reshape(input_image.shape[1]*3,input_image.shape[2]*3)
        var_tmp=computeTotalVariation(realign_reshape)
        var.append(var_tmp)
        tvMinIdx = np.argmin(var)
    return tvMinIdx,var

def computeTotalVariation(img):
    xKer = np.array([[-1,1]])
    yKer = xKer.transpose()
    tv = abs(signal.convolve2d(img,xKer)).sum()+ abs(signal.convolve2d(img,yKer)).sum()
    return tv

def undistort_model_load(center_pt):
    block_x,block_y = 7,5
    step_x, step_y = 1965,1965
    nx_1, ny_1 = 161,161
    nx_2, ny_2 = 159,159
    Nnum = 15
    if os.path.exists('./reconstruction/source/realign/sub_x_undistorted.npy') and os.path.exists('./reconstruction/source/realign/sub_y_undistorted.npy'):
        sub_x_undistorted = np.load('./reconstruction/source/realign/sub_x_undistorted.npy')
        sub_y_undistorted = np.load('./reconstruction/source/realign/sub_y_undistorted.npy')
    else:
        print('block y step:',step_y,', block x step:',step_x)
        center_x = np.arange(center_pt[1]-step_x*(block_x//2),center_pt[1]+step_x*(block_x//2)+1,step_x,dtype=int)
        center_y = np.arange(center_pt[0]-step_y*(block_y//2),center_pt[0]+step_y*(block_y//2)+1,step_y,dtype=int)
        print('block x center:', center_x,', block y center:',center_y)
        torch.cuda.synchronize()
        UndistortModel = glob.glob('./reconstruction/source/realign/undistort_params_dict_*.pkl')[0]
        with open(UndistortModel,'rb') as file:
            params=pickle.load(file)
        sub_y_undistorted = []
        sub_x_undistorted = []
        for i in range(block_y):
            for j in range(block_x):
                x_undistorted,y_undistorted = sub_undistort_coor(params, center_x[j], center_y[i], nx_2, ny_2)
                sub_start_y = center_y[i]-(Nnum*ny_1)//2
                sub_start_x = center_x[j]-(Nnum*nx_1)//2
                sub_x_undistorted.append(x_undistorted-sub_start_x)
                sub_y_undistorted.append(y_undistorted-sub_start_y)
        sub_x_undistorted = np.array(sub_x_undistorted, dtype=np.float32).reshape(len(sub_x_undistorted),ny_2*Nnum,nx_2*Nnum) # block_x*block_y, ny_2*Nnum, nx_2*Nnum
        sub_y_undistorted = np.array(sub_y_undistorted, dtype=np.float32).reshape(len(sub_y_undistorted),ny_2*Nnum,nx_2*Nnum)# block_x*block_y, ny_2*Nnum, nx_2*Nnum
        np.save('./reconstruction/source/realign/sub_x_undistorted.npy',sub_x_undistorted)
        np.save('./reconstruction/source/realign/sub_y_undistorted.npy',sub_y_undistorted)
    sub_x_undistorted = sub_x_undistorted/(nx_1*Nnum)*2-1
    sub_y_undistorted = sub_y_undistorted/(ny_1*Nnum)*2-1
    
    return sub_x_undistorted, sub_y_undistorted

def block_undistort(block_lf, sub_x_undistorted, sub_y_undistorted):
    '''
    input:
        lf_raw: (9,10748,14304), dtype=torch.tensor
    output:
        lf_undistort:(block_x*block_y, 9, ny_2*Nnum, nx_2*Nnum), dtype=torch.tensor
    '''
    grid = torch.stack([torch.from_numpy(sub_x_undistorted.astype(np.float32)), torch.from_numpy(sub_y_undistorted.astype(np.float32))], dim=2).unsqueeze(0)# 1,2100,2100,2
    
    block_lf_undistort = F.grid_sample(block_lf.unsqueeze(0), grid.to(block_lf.device), mode='bilinear', padding_mode='zeros', align_corners=False) # 90, ny_2*Nnum, nx_2*Nnum    
    return block_lf_undistort.squeeze()

def White_Balance(LF_wdf:torch.tensor,white_wdf:torch.tensor):
    '''
    LF_wdf: Wigner of LF after resize and realign, tensor(90, Nnum**2, ny_2, nx_2)
    White_wdf: Wigner of White image after realign,tensor(Nnum**2, ny_2, nx_2)
    '''
    # base = torch.mean(white_wdf, axis=(1,2), keepdims=True) # (400,1,1)
    # white = white_wdf/base
    # print(LF_wdf.shape)
    # print(white_wdf.shape)
    LF_wdf = LF_wdf/white_wdf.clamp(0.6,1.4) # 90,400,105,105

    return LF_wdf

def White_Balance_Debleaching(LF_wdf:torch.tensor,white_wdf:torch.tensor):
    '''
    LF_wdf: Wigner of LF after resize and realign, tensor(9, 81, ny_2, nx_2)
    White_wdf: Wigner of White image after realign,tensor(81, ny_2, nx_2)
    '''
    baseNorm = torch.sum(LF_wdf[0].unsqueeze(0),dim=[2,3],keepdims=True) # 1,81,1,1
    LF_wdf = LF_wdf / torch.sum(LF_wdf,dim=[2,3],keepdims=True)*baseNorm # 90,81,1,1
    LF_wdf = LF_wdf/white_wdf.to(LF_wdf.device).clamp(0.6,1.4) # 9, 81, ny_2, nx_2

    return LF_wdf


def realign_reshape(lf_undistort,white_img, Nnum=15):
    '''
    input:
        lf_undistort: (90, ny_2*Nnum, nx_2*Nnum), dtype=torch.tensor
        white_img: (Nnum**2, ny_2, nx_2), dtype=torch.tensor
    output:
        wigner: (90, Nnum**2, ny_2, nx_2), dtype=torch.tensor
    '''
    wigner = lf_undistort.reshape(lf_undistort.shape[0],lf_undistort.shape[-2]//Nnum,Nnum,lf_undistort.shape[-1]//Nnum,Nnum) \
        .permute(0,2,4,1,3).reshape(lf_undistort.shape[0],Nnum**2,lf_undistort.shape[-2]//Nnum,lf_undistort.shape[-1]//Nnum) # 90, Nnum**2, ny_2, nx_2
    baseNorm = torch.mean(wigner[0].unsqueeze(0),dim=[2,3],keepdims=True) # 1,400,1,1
    wigner = wigner / torch.mean(wigner,dim=[2,3],keepdims=True)*baseNorm # 90,400,1,1

    wigner_white = White_Balance(wigner, white_img.to(wigner.device))
    # wigner_white = remove_outliers(wigner)

    return wigner_white

def realign_merge(wigner_white, order, group_mode, start_frame, nshift=3):
    '''
    input:
        wigner: ( 90, Nnum**2, ny_2, nx_2), dtype=torch.tensor
        order: (3,3), dtype=np.array
    output:
        merge_wigner: (82(10), Nnum**2, ny_2*nshift, nx_2*nshift)
    '''
    order = order.flatten().tolist()
    n, v, h, w = wigner_white.shape
    if group_mode == 1:  ## n,225,135,135->n-8,225,405,405
        merged_wigner = torch.zeros(n - nshift**2 + 1, v, h*nshift, w*nshift, device=wigner_white.device)
        for i in range(merged_wigner.shape[0]):
            wigner_tmp = wigner_white[i:i + nshift**2]
            wigner_tmp = torch.roll(wigner_tmp, (i+start_frame) % nshift**2, 0)
            merged_wigner[i] = wigner_tmp[order].reshape(nshift, nshift, v, h, w).permute(2, 3, 0, 4, 1).reshape(v, h*nshift, w*nshift)
    else:  ## n,225,135,135-> n//9,225,405,405
        merged_wigner = torch.zeros(n//nshift**2, v, h*nshift, w*nshift, device=wigner_white.device)
        for i in range(merged_wigner.shape[0]):
            wigner_tmp = wigner_white[i * nshift**2:(i + 1) * nshift**2]
            merged_wigner[i] = wigner_tmp[order].reshape(nshift, nshift, v, h, w).permute(2, 3, 0, 4, 1).reshape(v, h*nshift, w*nshift)

    return merged_wigner


def remove_outliers(img, kernel_size=3, threshold=40):
    # 定义卷积核
    kernel = torch.ones(kernel_size, kernel_size) / kernel_size**2  # 定义一个3x3的均值滤波器
    for frame in range(img.shape[0]):
        for view in range(img.shape[1]):
            # 计算像素点与其周围点的均值
            img_conv = img[frame,view]
            mean_image = F.conv2d(img_conv.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).to(img_conv.device), padding=kernel_size//2)  # 使用2D卷积计算均值

            # 计算像素点与均值之间的差异
            diff = mean_image - img_conv  # 计算像素值相对于均值的差值

            # 去除离群值
            img[frame,view] = torch.where(diff > threshold, mean_image, img_conv)

    return img