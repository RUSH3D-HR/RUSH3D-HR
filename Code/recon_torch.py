# -*- coding: utf-8 -*-
'''
parallel recon(tensorrt) and save - memory load volume and stitch
'''
import os
from datetime import datetime
import argparse
import shutil

import numpy as np
import torch
from utils import *
import json
from undistort_realign.undistort_realign import White_Balance_Debleaching, block_undistort, realign_reshape, realign_merge, undistort_model_load
from lf_denoise.lf_denoise_main import lf_denoise_init, lf_denoise_infer
from lf_denoise.single_snr import single_snr
from VS_LFM.VS_LFM_init import init_vs_lfm
from VS_LFM.VS_LFM_infer import VS_LFM_inference, detect_artefact
import AIRecon
from block import BlockGenerator

from concurrent.futures import ThreadPoolExecutor
import queue
import concurrent.futures
import threading
import traceback
# import tensorrt as trt
# from torch2trt import TRTModule

def init(config_path):

    cur_root = os.getcwd().replace('\\','/') + '/reconstruction'
    scan_config_path = os.getcwd().replace('\\','/') + '/3x3.conf.sk.png'

    log_out("cur_root: %s" % cur_root)
    log_out("=============== INITIALIZATION ==============")
    with open(config_path, 'r') as f: config = json.load(f)

    # -----------------------------------project info init-----------------------------
    proj_name = config["proj_name"]

    input_folder, save_folder = config["input_folder"], config["save_folder"]
    if not os.path.exists(save_folder): os.makedirs(save_folder)

    sites = config["sites"]
    if sites is None or not isinstance(sites, list) or sites == []:
        raise Exception("the input argument 'sites' is invalid")
    site = sites[0]

    group_mode = config["group_mode"]
    if group_mode not in [0, 1]:
        raise Exception("the input argument 'group_mode' must = 0 / 1")
    is_full_img = config['is_full_img']
    channels = config["channels"]

    if channels is None or not isinstance(channels, list) or channels == []:
        raise Exception("the input argument 'channels' is invalid")
    
    for channel in channels:
        if not os.path.exists(save_folder + '/C' + str(channel)): os.makedirs(save_folder + '/C' + str(channel))
        if not os.path.exists(save_folder + '/C' + str(channel) + '_z32'): os.makedirs(save_folder + '/C' + str(channel) + '_z32')
        if not os.path.exists(save_folder + '/C' + str(channel) + '_MIP'): os.makedirs(save_folder + '/C' + str(channel) + '_MIP')

    start_frame, stop_frame = config["start_frame"], config["stop_frame"]

    if start_frame > stop_frame:
        raise Exception('start_frame must <= stop_frame')
    if len(channels) == 1 and config['skip_exist']:
        for frame in range(start_frame, stop_frame + 1):
                path = save_folder+f'/C{channels[0]}_MIP/{proj_name}_S{site}_C{channels[0]}_T{frame + 1}_mip.tif'
                if not os.path.exists(path):
                    break
        if frame != start_frame:
            log_out(f"Found last frame {frame-1}, setting start frame next to it")
            start_frame = frame - 1

    # -----------------------------------project info init-----------------------------

    # -----------------------------------realign init-----------------------------
    startX, startY = config["startX"], config["startY"]
    center_pt = [startY, startX]

    scan_config = get_scan_config(scan_config_path)
    sub_x_undistorted, sub_y_undistorted = undistort_model_load(center_pt)

    if is_full_img == 1:
        block_gen = BlockGenerator(center_pt=center_pt)
    else:
        block_gen = None

    if "grid" in config:
        grid = config['grid']
    else:
        grid = None
    # -----------------------------------realign init-----------------------------

    # -----------------------------------vsr model init-----------------------------
    if config["vsr_model"] is not None:
        log_out("VSR mode AUTO")
        VS_mode = 'AI'
        vsr_model_path = cur_root + f'/source/vsr_model/{config["vsr_model"]}_{VS_mode}.pth'
        vsr_generator_cfg = 'VS_LFM'
    else: 
        log_out("VSR mode OFF")
        vsr_model_path = None
        vsr_generator_cfg = 'VS_LFM'
    # -----------------------------------vsr model init-----------------------------

    # -----------------------------------denoise model init-----------------------------
    if config["denoise_model"] is not None:
        log_out("3X3 Denoise mode AUTO")
        denoise_model_path = cur_root + '/source/denoise_model/' + config["denoise_model"]
        log_out('denoise_model_path: %s' % denoise_model_path)
        denoise_thres = 0
    else:
        log_out("Denoise mode OFF")
        denoise_model_path = None
        denoise_thres = 0
    # -----------------------------------denoise model init-----------------------------

    # -----------------------------------recon_mode init-----------------------------
    recon_mode = "AI"
    if config['recon_trt']:
        recon_model_path = cur_root + "/source/recon_model/" + config["recon_model"] + '.engine'    
    else:
        recon_model_path = cur_root + "/source/recon_model/" + config["recon_model"] + '.pth' 
    
    psf = []
    psf_depth = 121
    log_out("Using PSF depth " + str(psf_depth))
    psf_offset = 0
    if "psf_offset" in config:
        psf_offset = config['psf_offset']
    for i in range(35):
        psf.append(get_psf_shift_pt(cur_root + f"/source/psf_{config['psf']}/{i}", depth=psf_depth, offset=psf_offset))
    input_views = get_input_views(recon_mode)
    
    white_img = []
    if "white" not in config:
        white = 'white_25'
    else:
        white = config['white']
    log_out(f'Using white image {white}')
    if white is not None:
        for i in range(35):
            white_img_path = cur_root + f'/source/realign/{white}/wigner_{i}.tif'
            white_img.append(tifffile.imread(white_img_path))
        white_img = torch.tensor(np.array(white_img, dtype=np.float32))
        base = torch.mean(white_img,axis=(0,2,3),keepdim=True)
        white_img = white_img / base
        white_img = white_img[:,input_views]
    else:
        white_img = None
    
    realign_only = config['realign_only']
    save_roi = config['save_roi']
    if save_roi is not None:
        assert len(save_roi) == 4
    save_mip_only = config['save_mip_only']
    # -----------------------------------recon_mode init-----------------------------
    remove_bg = config['remove_bg']
    save_multi = config['save_multi']
    # -----------------------------------batch init-----------------------------

    batch_id_list = []
    if group_mode == 0:
        wigner_id0, wigner_id1 = 9 * start_frame, 9 * stop_frame + 8
    else:
        wigner_id0, wigner_id1 = start_frame, stop_frame + 8
    total_wigner_num = wigner_id1 - wigner_id0 + 1
    batch_id_list = get_lf_ind_list(batch_id_list, group_mode, total_wigner_num, nshift=3)
    # -----------------------------------batch init-----------------------------
    
    log_out("========== INITIALIZATION FINISHED ==========\n")
    now_str = datetime.now().strftime("%Y-%d-%m-%H-%M-%S")
    shutil.copy(config_path, save_folder + f'/{now_str}_config.json')

    return {"proj_name": proj_name, 
            "input_folder": input_folder, 
            "save_folder": save_folder, 
            "save_mip_only": save_mip_only,
            "save_roi": save_roi,
            "site": site, 
            "group_mode": group_mode, 
            "is_full_img": is_full_img,
            "block_gen": block_gen,
            "channels": channels, 
            "start_frame": start_frame, 
            "stop_frame": stop_frame, 
            "sub_x_undistorted": sub_x_undistorted, 
            "sub_y_undistorted": sub_y_undistorted, 
            "grid": grid,
            "white_img": white_img, 
            "input_views": input_views, 
            "wigner_id0": wigner_id0, 
            "wigner_id1": wigner_id1, 
            "batch_id_list": batch_id_list, 
            "scan_config": scan_config, 
            "denoise_model_path": denoise_model_path, 
            "denoise_thres": denoise_thres,
            "vsr_model_path": vsr_model_path, 
            "vsr_generator_cfg": vsr_generator_cfg, 
            'realign_only': realign_only,
            "psf": psf, 
            "recon_mode": recon_mode,
            "recon_model_path": recon_model_path,
            "remove_bg": remove_bg,
            "save_multi": save_multi}

def preprocess(config, block, device, channel, batch_id, frame_count, lock=None, batch_lf_stack=None):
    global preprocess_buffer
    if batch_lf_stack is None:
        batch_lf_stack = get_lf_list_2(config["input_folder"], config["proj_name"], config["site"], channel, block+1, batch_id, config["wigner_id0"])
    if lock is not None:
        lock.acquire()
    batch_lf_stack = torch.from_numpy(batch_lf_stack).to(device)
    log_out('----------start to process B%d T:%d-%d----------' % (block, config["wigner_id0"]+batch_id[0] + 1, config["wigner_id0"]+batch_id[1]))

    t = time()
    batch_lf_stack = block_undistort(batch_lf_stack, config["sub_x_undistorted"][block], config["sub_y_undistorted"][block])
    log_out('block-undistort takes: %.5f s' % (time() - t))
    # save_block_undistort_path = get_block_undistort_path(config["save_folder"], config["proj_name"], config["site"], channel, block+1, config["start_frame"] + frame_count + 1)
    # add_save_task(buffer, save_block_undistort_path, batch_lf_stack[0].to(torch.int16).cpu().numpy().astype(np.uint16))

    t = time()
    batch_lf_stack = realign_reshape(batch_lf_stack, config["white_img"][block])
    log_out('realign-reshape takes: %.5f s' % (time() - t))
    # save_realign_reshape_path = get_realign_reshape_path(config["save_folder"], config["proj_name"], config["site"], channel, block+1, config["start_frame"] + frame_count + 1)
    # add_save_task(buffer, save_realign_reshape_path, batch_lf_stack[0].to(torch.int16).cpu().numpy().astype(np.uint16))

    batch_lf_stack = batch_lf_stack[:, config["input_views"], :, :].cpu() # 90, 81, 159, 159
    if lock is not None:
        lock.release()
    # batch_lf_stack = batch_lf_stack.cpu() # 90, 81, 159, 159 # SAVE_REALIGN_ONLY

    # add_save_task(buffer, save_realign_reshape_path.replace('.tif', '_batch_CV.tif'),
                # (batch_lf_stack[8:,0,:,:]if config["group_mode"] == 1 else batch_lf_stack[4:-4: 9,0,:,:]).numpy().clip(0, 2 ** 16 - 1).astype(np.uint16))
    preprocess_buffer.put([block,batch_lf_stack])
    del batch_lf_stack
    torch.cuda.empty_cache()

def preprocess_grid(batch_lf_stack, config, device, channel, frame_count, grid=None):
    global preprocess_buffer
    """
    batch_lf_stack is a pytorch, [90, [81], H, W]
    """
    if batch_lf_stack.ndim == 3:
        if config['grid'] is None:
            grid = torch.load(f'./reconstruction/source/realign/grid.pt')
        else:
            log_out(f"Using grid {config['grid']}")
            grid = torch.load(f'./reconstruction/source/realign/{config["grid"]}.pt')
        # grid shape is 1, 10245, 14175, 2
        tt = time()

        dis_batch = torch.zeros(batch_lf_stack.shape[0],grid.shape[1], grid.shape[2]) # 90,10245,14175
        for t in range(batch_lf_stack.shape[0]):
            dis_batch[t] = torch.nn.functional.grid_sample(batch_lf_stack[t].to(device).unsqueeze(0).unsqueeze(0), grid.to(device), align_corners=True).squeeze(0).squeeze(0)
        
        batch_lf_stack = dis_batch
        del dis_batch

        # batch_lf_stack = torch.nn.functional.grid_sample(batch_lf_stack.cuda().unsqueeze(0), grid.cuda(), align_corners=True).squeeze(0) # 90,10245,14175
        print(f"Grid calibration takes {time() - tt}")
        batch_lf_stack = batch_lf_stack.reshape(batch_lf_stack.shape[0], 683, 15, 945, 15).permute(0, 2, 4, 1, 3).reshape(batch_lf_stack.shape[0], 225, 683, 945)
        # save_realign_reshape_path = get_realign_reshape_path(config["save_folder"], config["proj_name"], config["site"], channel, 0, config["start_frame"] + frame_count + 1)
        # add_save_task(buffer, save_realign_reshape_path, batch_lf_stack[0].to(torch.int16).cpu().numpy().astype(np.uint16))
        # if True:
        if not config['realign_only']:
            batch_lf_stack = batch_lf_stack[:,config["input_views"],:,:].cpu()
        else:
            batch_lf_stack = batch_lf_stack.cpu()
        
    for block_y in range(5):
        for block_x in range(7):
            block = block_y*7+block_x
            start_x = block_x*131
            start_y = block_y*131
            # t = time()
            block_batch_lf_stack = batch_lf_stack[:, :, start_y:start_y+159, start_x:start_x+159]
            # block_batch_lf_stack = batch_lf_stack[:, :, start_y:start_y+159, start_x:start_x+159].to(device)
            # block_batch_lf_stack = White_Balance(block_batch_lf_stack, config['white_img'][block].to(device))
            # log_out("Block white balance takes " + str(time() - t))
            preprocess_buffer.put([block,block_batch_lf_stack])
    del batch_lf_stack
    torch.cuda.empty_cache()
    # gc.collect()

def recon_one_frame(config, batch_lf_stack, block, device, wigner_idx, batch_id, frame_count, lock=None):
    '''
    batch_lf_stack: tensor(9, 81, 159, 159)]
    '''
    global recon_buffer
    torch.cuda.set_device(device)

    if lock is not None:
        lock.acquire()

    torch.cuda.empty_cache()
    log_out('----------start to merge and recon B%d T%d----------' % (block, config["start_frame"] + frame_count + 1))

    t = time()
    if (not config['realign_only']) and (config['white_img'] is not None):
        batch_lf_stack = White_Balance_Debleaching(batch_lf_stack, config["white_img"][block])
    batch_lf_stack = realign_merge(batch_lf_stack, config["scan_config"], config["group_mode"],  config["wigner_id0"] + batch_id[0] + wigner_idx) # 1, 81, 477, 477
    # torch.cuda.synchronize()
    log_out('realign-merge takes: %.5f s' % (time() - t))
    if config['realign_only']:
        save_realign_merge_path = get_realign_merge_path(config["save_folder"], config["proj_name"], config["site"], channel, block+1, config["start_frame"] + frame_count + 1)
        add_save_task(buffer, save_realign_merge_path, batch_lf_stack[0].to(torch.int16).cpu().numpy().astype(np.uint16))
        return None # SAVE_REALIGN_ONLY
    # batch_lf_stack *= 0.2
    # VS_LFM
    if config["vsr_model_path"] is not None and detect_artefact(batch_lf_stack, device) > 0.3:
        t = time()
        vsr_model = init_vs_lfm(config["vsr_model_path"], device)
        wigner_id0_vs = config["wigner_id0"] + batch_id[0] + wigner_idx + 8 if config["group_mode"] == 1 else config["wigner_id0"] + batch_id[0] + wigner_idx + 4
        batch_lf_stack, _ = VS_LFM_inference(vsr_model, batch_lf_stack, config["scan_config"], wigner_id0_vs, device)
        del vsr_model
        torch.cuda.empty_cache()
        log_out('VSR takes: %.5f s' % (time() - t))
    else:
        log_out('No motion, No need to VSR')
        if config["vsr_model_path"] is not None:
            config["vsr_model_path"] = None
    if config['realign_only']:
        return None
    
    if config["denoise_model_path"] is not None and single_snr(batch_lf_stack[0, 0]) < 15:
        thres = batch_lf_stack[0].std() * 2
        t = time()
        denoise_generator, fusion_layer = lf_denoise_init(config["denoise_model_path"], device)
        batch_lf_stack_high = torch.where(batch_lf_stack >= thres, batch_lf_stack, 0).to(device)
        batch_lf_stack_low = torch.where(batch_lf_stack < thres, batch_lf_stack, 0).to(device)

        batch_lf_stack_low = lf_denoise_infer(batch_lf_stack_low, denoise_generator,  fusion_layer, device)
        batch_lf_stack_low = batch_lf_stack_low.to(device)
        batch_lf_stack = batch_lf_stack_low + batch_lf_stack_high
        batch_lf_stack = batch_lf_stack.clamp(0, 2 ** 32 - 1)
        torch.cuda.empty_cache()
        log_out('denoise takes: %.5f s' % (time() - t))
        if config['realign_only']:
            save_denoise_path = get_denoise_path(config["save_folder"], config["proj_name"], config["site"], channel, block+1, config["start_frame"] + frame_count + 1)
            add_save_task(buffer, save_denoise_path, batch_lf_stack[0].cpu().numpy().clip(0, 2 ** 16 - 1).astype(np.uint16))
    else:
        log_out('No need to denoise')
        if config["denoise_model_path"] is not None:
            config["denoise_model_path"] = None
    
    lf_stack = batch_lf_stack.squeeze(0).to(device)

    psf_block = config["psf"][block].to(device)

    alpha = torch.linspace(0, 1, 50)
    transition = (1 / (1 + torch.exp(-10 * (alpha - 0.5)))).to(device)
    t = time()
    # model_weight = torch.load(config['recon_model_path'])['model']
    # recon_model = AIRecon.make(model_weight, load_sd=True).cuda()
    # recon_model.eval()

    global recon_model_list
    recon_model = recon_model_list[device]

    lf_stack -= config['remove_bg']
    lf_stack = torch.clamp(lf_stack, min=0)
    recon_img = recon_model(lf_stack, psf_block, transition).squeeze()
    recon_img *= config['save_multi']
    torch.cuda.synchronize()
    log_out('AI recon takes: %.5f s ' % (time() - t))
    
    t = time()
    global ms_grid
    recon_img = recon_post_process(recon_img, target_z=91)
    log_out('Post-process takes: %.5f s ' % (time() - t))


    recon_buffer.put([block,recon_img])
    del lf_stack, recon_img, recon_model
    torch.cuda.empty_cache()

    # gc.collect()
    if lock is not None:
        lock.release()

    # log_out('-------------------------------------------\n')
    # return recon_img

# pyinstaller --version-file file_version_info.txt ./Code/recon.py
if __name__ == '__main__':
    log_out("=============== PROGRAM BEGINS AT %s =============== \n" % (datetime.now()))
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6'

    if not torch.cuda.is_available():
        raise Exception('Error: No available GPU device')
    else:
        try:
            ## load config file
            parser = argparse.ArgumentParser()
            parser.add_argument('--config', default='./RCConfig.json')
            args = parser.parse_args()
            config = init(args.config)
            n_gpu = torch.cuda.device_count()
            log_out("\n>>> GPU AVAILABLE %d" % (n_gpu))
                
            ## initialize saving threadings
            buffer = queue.Queue(maxsize=(7+4+2)*3)
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
            exit_flag = threading.Event()
            consumer_futures = {executor.submit(saving_task, buffer, exit_flag, f"Consumer-{i}"): i for i in range(7)}

            gpu_lock_list = []
            for i in range(n_gpu):
                lock = threading.Lock()
                gpu_lock_list.append(lock)

            global recon_model_list
            recon_model_list = {}
            if config["recon_mode"] == "AI":
                for i in range(n_gpu):
                    recon_model_path = config['recon_model_path']
                    torch.cuda.set_device(i)
                    if recon_model_path.endswith('.pth'):
                        model_weight = torch.load(config['recon_model_path'])['model']
                        recon_model = AIRecon.make(model_weight, load_sd=True).to(f'cuda:{i}')
                        recon_model.eval()
                    else:
                        raise NotImplementedError("Please install tensorrt to use trt models")
                        # logger = trt.Logger(trt.Logger.INFO)
                        # with open(config["recon_model_path"], "rb") as f, trt.Runtime(logger) as runtime:
                        #     recon_model = runtime.deserialize_cuda_engine(f.read())
                        # recon_model = TRTModule(recon_model, input_names=['input','psf', 'transition'], output_names=['output'])
                    recon_model_list[f'cuda:{i}'] = recon_model
            
            t_start = time()
            for channel in config["channels"]:
                log_out("\n>>> START TO RECONSTRUCT C%d" % (channel))
                frame_count = 0
                for batch_id in config["batch_id_list"]:
                    # gc.collect()
                    # end exisitance test

                    t = time()
                    preprocess_buffer = queue.Queue(maxsize=35)
                    is_full_img = config["is_full_img"]

                    if is_full_img == 0:
                        with ThreadPoolExecutor(max_workers=35) as executor:
                            preprocess_future = [executor.submit(preprocess, config, block, 'cuda:%d'%(block%n_gpu), channel, batch_id, frame_count, gpu_lock_list[block % n_gpu]) for block in range(35)]
                        concurrent.futures.wait(preprocess_future)
                        for future in concurrent.futures.as_completed(preprocess_future):
                            try:
                                preprocess_future.remove(future)
                                result = future.result()
                            except Exception as e:
                                print("Error:", e)
                                raise e
                    elif is_full_img >= 1:
                        batch_lf_stack, _ = get_lf_list_full_grid(config["input_folder"], config["proj_name"], config["site"], channel, batch_id, config["wigner_id0"])
                        preprocess_grid(batch_lf_stack, config, 'cuda:0', channel, frame_count)
                        del batch_lf_stack
                    else:
                        raise NotImplementedError

                    log_out("\n=============== PREPROCESS TAKES: %.1f s================" % (time() - t))
                    
                    block = []
                    batch_lf_stack = [] # 35, 90, 81, 159, 159
                    while not preprocess_buffer.empty():
                        index, preprocess_img = preprocess_buffer.get()
                        block.append(index)
                        batch_lf_stack.append(preprocess_img)
                        preprocess_buffer.task_done()
                    del preprocess_buffer
                    batch_lf_stack = [batch_lf_stack[block.index(i)] for i in range(len(block))]
                    batch_size = batch_lf_stack[0].shape[0]-8 if config["group_mode"] == 1 else int(batch_lf_stack[0].shape[0]/9)
                    recon_buffer = queue.Queue(maxsize=35)

                    torch.cuda.empty_cache()

                    # tracemalloc.start()
                    for frame in range(batch_size):
                        if config["group_mode"] == 0:
                            wigner_idx = 9 * frame
                        else:
                            wigner_idx = frame
                        t = time()
                        
                        with ThreadPoolExecutor(max_workers=n_gpu) as executor:
                            recon_futures = [executor.submit(recon_one_frame, config, batch_lf_stack[block][wigner_idx:wigner_idx+9].to('cuda:%d'%(block%n_gpu)),
                                                             block, 'cuda:%d'%(block%n_gpu), wigner_idx, batch_id, frame_count) for block in range(35)]
                        concurrent.futures.wait(recon_futures)
                        for future in concurrent.futures.as_completed(recon_futures):
                            try:
                                result = future.result()
                            except Exception as e:
                                print("Error:", e)
                        # for block in range(35):
                        #     recon_one_frame(config, batch_lf_stack[block][wigner_idx:wigner_idx+9].to('cuda:%d'%(block%n_gpu)),
                        #                                       block, 'cuda:%d'%(block%n_gpu), wigner_idx, batch_id, frame_count)

                        # gc.collect()
                        frame_count += 1
                        if config['realign_only']:
                            continue # SAVE_REALIGN_ONLY

                        block = []
                        volume_stack = []
                        tic_get_recon_res = time()
                        while not recon_buffer.empty():
                            index, volume = recon_buffer.get()
                            block.append(index)
                            volume_stack.append(volume)
                            recon_buffer.task_done()
                        
                        depth_batch = 9
                        volume_stack = [volume_stack[block.index(i)] for i in range(len(block))]
                        log_out("\n=============== FRAME %d RECONSTRUCTION TAKES: %.1f s================" % (config["start_frame"] + frame_count, time() - t))

                        tic = time()
                        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> parrallel trans list to numpy <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                        volume_stack_thread = threading_copy(volume_stack, num_threads=35)
                        del volume_stack

                        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> parrallel trans list to numpy <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                        log_out(f'recon_list to numpy cost time:{time() - tic}')

                        # sigmoid stitch
                        tic = time()
                        if config['recon_mode'] == "AI":
                            panorama = stitch_recon_result_v4(volume_stack_thread, z=None)
                        else:
                            pass
                        del volume_stack_thread
                        log_out(f'stitch cost time:{time()-tic}')

                        # SAVE HERE
                        if config['save_roi'] is not None:
                            x0, y0, x1, y1 = config['save_roi']
                            panorama = panorama[:, y0:y1, x0:x1]

                        mip = np.max(panorama, axis=0).astype(np.uint16)
                        add_save_task(buffer, config["save_folder"]+f'/C{channel}_MIP/{config["proj_name"]}_S{config["site"]}_C{channel}_T{config["start_frame"]+frame_count}_mip.tif', mip)
                        
                        if not config['save_mip_only']:
                            if panorama.shape[0] == 91:
                                panorama = panorama[1:]
                            for i in range(int(np.ceil(panorama.shape[0]/depth_batch))):
                                add_save_task(buffer, config["save_folder"]+f'/C{channel}/{config["proj_name"]}_S{config["site"]}_C{channel}_T{config["start_frame"]+frame_count}_{i}.tif',panorama[i*depth_batch:(i+1)*depth_batch])
                        del panorama
                        log_out('\nProcess: [Current site:%d, Current channel:%d, Current frame:%d, Total frame:%d]' % (config["site"], channel, frame_count, config["stop_frame"]-config["start_frame"]+ 1))
                    del batch_lf_stack
            log_out("=============== RECONSTRUCTION FINISHED AT %s total: %.1f s================" % (datetime.now(), time() - t_start))

            set_finish(consumer_futures, exit_flag)

            log_out("=============== SAVE FINISHED AT %s total: %.1f s================\n\n\n\n\n\n" % (datetime.now(), time() - t_start))
                
        except Exception as e:
            log_out(str(e))
            log_out(traceback.format_exc())
            exit_flag.set()
            # set_finish(consumer_futures, exit_flag)
            concurrent.futures.wait(consumer_futures)
            raise Exception("RECONSTRUCTION STOP, TERMINATING SAVING THREAD\n\n\n\n\n\n")
        finally:
            # recon_executor.shutdown()
            executor.shutdown()
