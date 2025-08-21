"""
Test the LF-denoising model
The Code is created based on the method described in the following paper
    ZHI LU, WENTAO CHEN etc.
    Self-supervised light-field denoising empowers high-sensitivity fast 3D fluorescence imaging without
    temporal dependency
    Submitted, 2024.
    Contact: ZHI LU (luzhi@tsinghua.edu.cn)
"""

import math
import argparse
import os

import numpy as np
import torch.cuda
import time

from lf_denoise.sub_network import SRDTrans
from lf_denoise.lf_denoising import LFDenoising, FusionModule
from lf_denoise.utils.sampling import *
from lf_denoise.utils.data_process import LF_2_buck, im_resize, test_preprocess, testset, singlebatch_test_save, \
    multibatch_test_save
from torch.utils.data import DataLoader
from lf_denoise.utils.round_view import RoundView

# init parameters
opt = argparse.Namespace()
opt.patch_s = 256
opt.total_view = 225
opt.overlap_factor = 0.125
opt.batch_size = 1
opt.radius = 5
opt.scale_factor = 1
opt.cut_uv = 2
opt.upscale = 1
opt.export_uint16 = True
one_side_total_view = np.sqrt(opt.total_view)
assert one_side_total_view ** 2 == opt.total_view
# use isotropic patch size by default
rv = RoundView(opt.radius, total_view=int(one_side_total_view))


def lf_denoise_init(model_path, device):
    # for u-net compatibility, patch_y is the dimension of selected views
    opt.patch_y = rv.size
    opt.patch_x = opt.patch_s
    opt.patch_t = opt.patch_s

    opt.gap_t = int(opt.patch_t * (1 - opt.overlap_factor))
    opt.gap_x = int(opt.patch_x * (1 - opt.overlap_factor))
    opt.gap_y = int(opt.patch_y * (1 - opt.overlap_factor))

    # denoise_generator = LFDenoising(
    #     img_dim=opt.patch_x,
    #     img_time=opt.patch_t,
    #     in_channel=1,
    #     embedding_dim=128,
    #     num_heads=8,
    #     hidden_dim=128 * 4,
    #     lr=1e-5,
    #     b1=0.5,
    #     b2=0.999,
    #     window_size=7,
    #     num_transBlock=1,
    #     attn_dropout_rate=0.1,
    #     f_maps=[16, 32, 64],
    #     input_dropout_rate=0,
    #     device=device
    # )
    denoise_generator_1 = SRDTrans(
        img_dim=opt.patch_x,
        img_time=opt.patch_t,
        in_channel=1,
        embedding_dim=128,
        num_heads=8,
        hidden_dim=128 * 4,
        window_size=7,
        num_transBlock=1,
        attn_dropout_rate=0.1,
        f_maps=[16, 32, 64],
        input_dropout_rate=0,
    ).to(device)
    denoise_generator_2 = SRDTrans(
        img_dim=opt.patch_x,
        img_time=opt.patch_t,
        in_channel=1,
        embedding_dim=128,
        num_heads=8,
        hidden_dim=128 * 4,
        window_size=7,
        num_transBlock=1,
        attn_dropout_rate=0.1,
        f_maps=[16, 32, 64],
        input_dropout_rate=0,
    ).to(device)

    fusion_layer = FusionModule(inplanes=2 * opt.patch_y, planes=opt.patch_y)

    # load model
    denoise_generator_1.load_state_dict(torch.load(model_path+'_0.pth'))
    denoise_generator_2.load_state_dict(torch.load(model_path+'_1.pth'))
    fusion_layer.load_state_dict(torch.load(model_path + "_fl.pth", map_location="cpu"))
    # denoise_generator.train(False)
    fusion_layer.train(False)

    fusion_layer.to(device)

    return denoise_generator_1, denoise_generator_2, fusion_layer


def trans_torch_2_onnx(model, input_shape, output_path,model_name = 'denoise'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    input_names = ["input_1", "input_2"]
    output_names = ["output"]
    input1 = torch.randn(input_shape).cuda()
    input2 = torch.randn(input_shape).cuda()
    if model_name == 'denoise':
        torch.onnx.export(model, input1, output_path, input_names=[input_names[0]], output_names=output_names,
                          opset_version=16)
    else:
        torch.onnx.export(model, (input1, input2), output_path, input_names=input_names, output_names=output_names, opset_version=16)

if __name__ == '__main__':
    model_path = r"F:\Code\trans_lf_denoise_2_trt\pth\E_30_Iter_1800"
    denoise_generator_1,denoise_generator_2, fusion_layer = lf_denoise_init(model_path,'cuda')
    denoise_input1_shape = (1, 1, opt.patch_s, 81, opt.patch_s)
    fusion_input1_shape = (1, 81, opt.patch_s, opt.patch_s)
    trans_torch_2_onnx(denoise_generator_1, denoise_input1_shape, output_path=model_path+'_1.onnx')
    trans_torch_2_onnx(denoise_generator_2, denoise_input1_shape, output_path=model_path+'_2.onnx')
    trans_torch_2_onnx(fusion_layer, fusion_input1_shape, output_path=model_path+'_fl.onnx', model_name = 'fusion')


