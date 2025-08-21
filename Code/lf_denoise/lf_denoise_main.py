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
import numpy as np
import torch.cuda
import time

from utils import log_out
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

    denoise_generator = LFDenoising(
        img_dim=opt.patch_x,
        img_time=opt.patch_t,
        in_channel=1,
        embedding_dim=128,
        num_heads=8,
        hidden_dim=128 * 4,
        lr=1e-5,
        b1=0.5,
        b2=0.999,
        window_size=7,
        num_transBlock=1,
        attn_dropout_rate=0.1,
        f_maps=[16, 32, 64],
        input_dropout_rate=0,
        device=device
    )

    fusion_layer = FusionModule(inplanes=2 * opt.patch_y, planes=opt.patch_y,use_residual=True)

    # load model
    denoise_generator.load(model_path)
    fusion_layer.load_state_dict(torch.load(model_path + "_fl.pth", map_location="cpu"))
    denoise_generator.train(False)
    fusion_layer.train(False)

    fusion_layer.to(device)

    return denoise_generator, fusion_layer


def lf_denoise_infer(src_img, denoise_generator, fusion_layer, device):
    log_out("Start denoising...")
    denoised_wigner = torch.empty(src_img.shape).permute(0,2,1,3)
    progress_bar = {}
    progress_count = 0
    src_img = src_img.to(device)
    tstart = time.time()
    for N in range(src_img.shape[0]):
        name_list, noise_img, coordinate_list = test_preprocess(src_img[N], opt, rv)  # noise_img:(h,n,w),n已经是cycle的视角索引

        test_data = testset(name_list, coordinate_list, noise_img, rv)
        testloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)
        total_task_num = src_img.shape[0] * len(name_list)
        if N == 0:
            progress_bar.update(
                {str(math.floor(0.25 * total_task_num)): '25%',
                 str(math.floor(0.5 * total_task_num)): '50%',
                 str(math.floor(0.75 * total_task_num)): '75%',
                 str(total_task_num): '100%'}
            )

        infer_time_cost = 0
        with (torch.no_grad()):
            for iteration, (xs_volume, yt_volume, single_coordinate) in enumerate(testloader):
                # torch.cuda.synchronize()
                tic = time.time()
                xs_predict, yt_predict = denoise_generator.forward([xs_volume, yt_volume])#input:[1,1,128,81,128],output:[1,1,128,81,128]
                # torch.cuda.synchronize()
                toc = time.time()
                # log_out(f"conv time cost: {toc-tic}")
                infer_time_cost += (toc - tic)
                xs_predict = torch.squeeze(xs_predict, 1)
                xs_predict_from_yt = torch.squeeze(yt_predict, 1).permute(0, 3, 2, 1)

                xs_predict_from_yt = xs_predict_from_yt[:, :, rv.cxt_2_cxs_index_list, :]

                xs_predict = xs_predict.permute(0, 2, 1, 3)
                xs_predict_from_yt = xs_predict_from_yt.permute(0, 2, 1, 3)

                # torch.cuda.synchronize()
                tic = time.time()
                predict = fusion_layer(xs_predict, xs_predict_from_yt)#input:[1,81,128,128],output:[1,81,128,128]
                # torch.cuda.synchronize()
                toc = time.time()
                # log_out(f"fusion time cost: {toc-tic}")
                infer_time_cost += (toc - tic)
                predict = predict.permute(0, 2, 1, 3)

                # output_image = np.squeeze(predict.cpu().detach().numpy())
                #
                # raw_image = np.squeeze(xs_volume.cpu().detach().numpy())

                output_image = np.squeeze(predict.cpu().detach())

                raw_image = np.squeeze(xs_volume.cpu().detach())

                if output_image.ndim == 3:
                    stack_cnt = 1
                else:
                    stack_cnt = output_image.shape[0]

                if stack_cnt == 1:
                    stack_img, _, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = \
                        singlebatch_test_save(single_coordinate, output_image, raw_image)
                    denoised_wigner[N, stack_start_s:stack_end_s, stack_start_h:stack_end_h,stack_start_w:stack_end_w] = stack_img
                else:
                    for i in range(stack_cnt):
                        stack_img, _, stack_start_w, stack_end_w, stack_start_h, stack_end_h, stack_start_s, stack_end_s = \
                            multibatch_test_save(single_coordinate, i, output_image, raw_image)
                        denoised_wigner[N, stack_start_s:stack_end_s, stack_start_h:stack_end_h,stack_start_w:stack_end_w] = stack_img

                progress_count += 1
                # if str(progress_count) in progress_bar.keys():
                #     log_out('--denoise progress: %s' % (progress_bar[str(progress_count)]))

            del noise_img
        # print("only infer time cost: %.3f" % infer_time_cost)
    # log_out("Denoising cost %.3f" % (time.time() - tstart))
    output_img = denoised_wigner * opt.scale_factor
    # output_img = output_img.transpose(
    #     (1, 0, 2))[rv.cxs_2_o_index_list]
    #
    # output_img = rv.rv_stack_2_lf(output_img)
    output_img = output_img.permute(0, 2, 1, 3)

    return output_img
