import copy

import torch
from torch2trt import TRTModule
import tensorrt as trt
import argparse
from time import time
import numpy as np
import math
import cv2
from lf_denoise.utils.sampling import *
from lf_denoise.utils.data_process import LF_2_buck, im_resize, test_preprocess, testset, singlebatch_test_save, \
    multibatch_test_save
from torch.utils.data import DataLoader
from lf_denoise.utils.round_view import RoundView
from lf_denoise.lf_denoising import LFDenoising, FusionModule
from utils import log_out

opt = argparse.Namespace()
opt.patch_s = 128
opt.total_view = 225
opt.overlap_factor = 0.25
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
opt.patch_y = rv.size
opt.patch_x = opt.patch_s
opt.patch_t = opt.patch_s

opt.gap_t = int(opt.patch_t * (1 - opt.overlap_factor))
opt.gap_x = int(opt.patch_x * (1 - opt.overlap_factor))
opt.gap_y = int(opt.patch_y * (1 - opt.overlap_factor))

def lf_denoise_trt_init(model_path,device) :

    torch.cuda.set_device(device)
    denoise_generator_1 = init_trt_model(model_path+'_0.engine',model_mode='denoise')
    denoise_generator_2 = init_trt_model(model_path+'_1.engine',model_mode='denoise')
    fusion_layer = init_trt_model(model_path+'_fl.engine',model_mode='fusion')

    return denoise_generator_1, denoise_generator_2, fusion_layer

def init_trt_model(model_path,model_mode='denoise'):
    torch.cuda.synchronize()
    trt_init_start = time()
    logger = trt.Logger(trt.Logger.INFO)
    input_names = ["input_1", "input_2"] if model_mode != 'denoise' else ["input_1"]
    with open(model_path, "rb") as f, trt.Runtime(
            logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
    model = TRTModule(model, input_names=input_names, output_names=['output'])
    torch.cuda.synchronize()
    trt_init_end = time()
    log_out(f"{model_mode} TRT init time: {(trt_init_end - trt_init_start):.4f}s")
    return model


def lf_denoise_trt_infer(src_img, denoise_generator_1, denoise_generator_2, fusion_layer, device):
    log_out("Start denoising...")
    init_time = time()
    denoised_wigner = torch.empty(src_img.shape).permute(0,2,1,3).to(device)
    progress_bar = {}
    progress_count = 0
    src_img = src_img.to(device)
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
                tic = time()
                xs_predict, yt_predict = denoise_generator_1(xs_volume), denoise_generator_2(yt_volume)#input:[1,1,128,81,128],output:[1,1,128,81,128]
                # torch.cuda.synchronize()
                toc = time()
                infer_time_cost += (toc - tic)

                xs_predict = torch.squeeze(xs_predict, 1)
                xs_predict_from_yt = torch.squeeze(yt_predict, 1).permute(0, 3, 2, 1)

                xs_predict_from_yt = xs_predict_from_yt[:, :, rv.cxt_2_cxs_index_list, :]
                #permute 导致tensor不连续，因此需要重新排列为连续的形式，防止trt推理时.contiguous().data_ptr()使用同一地址，导致先输入被后输入覆盖
                xs_predict = xs_predict.permute(0, 2, 1, 3)
                xs_predict = xs_predict.contiguous()
                xs_predict_from_yt = xs_predict_from_yt.permute(0, 2, 1, 3)
                xs_predict_from_yt = xs_predict_from_yt.contiguous()

                # torch.cuda.synchronize()
                tic = time()
                predict = fusion_layer(xs_predict,xs_predict_from_yt)#input:[1,81,128,128],output:[1,81,128,128]
                # torch.cuda.synchronize()
                toc = time()
                infer_time_cost += (toc - tic)
                predict = predict.permute(0, 2, 1, 3)

                # output_image = np.squeeze(predict.cpu().detach())
                output_image = predict.squeeze()
                # raw_image = np.squeeze(xs_volume.cpu().detach())
                raw_image = xs_volume.squeeze()

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
                if str(progress_count) in progress_bar.keys():
                    log_out('--denoise progress: %s' % (progress_bar[str(progress_count)]))

            del noise_img
        # log_out("only infer time cost: %.3f" % infer_time_cost)
    output_img = denoised_wigner * opt.scale_factor

    #no need to rearrange the order
    # output_img = output_img.transpose(
    #     (1, 0, 2))[rv.cxs_2_o_index_list]
    #
    # output_img = rv.rv_stack_2_lf(output_img)

    output_img = output_img.permute(0, 2, 1, 3)
    log_out("denoising time cost: %.3f" % (time() - init_time))
    return output_img

if __name__ == '__main__':
    import tifffile as tiff

    denoise_generator_1 = init_trt_model(r'/pth/mouse_liver_c4_0.engine')
    denoise_generator_2 = init_trt_model(r'/pth/mouse_liver_c4_1.engine')
    fusion_layer = init_trt_model(r'F:\Code\trans_lf_denoise_2_trt\pth\fusion_layer_sparsity.engine',model_mode='fusion')

    test_input = tiff.imread(r'Y:\2_Data\LQ\RUSH3D\test_denoise_recon\test_pure_C2_denoised\realign\Mouse_vessels_488_20X_S1_C2_B33_T1_realign_merge.tif').astype(np.float32)
    test_input = torch.from_numpy(test_input).unsqueeze(0)
    out_trt = lf_denoise_trt_infer(test_input, denoise_generator_1, denoise_generator_2, fusion_layer, torch.device('cuda:0'))
    out_trt = out_trt.squeeze().cpu().detach().numpy().astype(np.uint16)
    tiff.imwrite('test_out_wrong_same.tif', out_trt)