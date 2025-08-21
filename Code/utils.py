import gc
import os
import sys
import h5py
import torch
import tifffile
import queue
import threading
import numpy as np
from PIL import Image
from time import time, sleep
from datetime import datetime
import concurrent.futures

def stitch_recon_result_v4(recon_res,z=73,num_y=5,num_x=7,overlap=50):
    '''

    Args:
        recon_res: 35 block recon res: (35,63,H,W)
        edge:[35*ndarray:lrud]

    Returns: stitched recon res

    '''
    if z is None:
        z = recon_res.shape[1]
    cut_off = overlap//2
    yx, d, h, w = recon_res.shape
    recon_res = recon_res.reshape(num_y, num_x, d, h, w)
    tic = time()
    left_edge = threading_copy(recon_res[:, :, :, :, :overlap],axis=2, num_threads=z)
    right_edge = threading_copy(recon_res[:, :, :, :, -overlap:],axis=2, num_threads=z)
    up_edge = threading_copy(recon_res[:, :, :, :overlap, :],axis=2, num_threads=z)
    down_edge = threading_copy(recon_res[:, :, :, -overlap:, :],axis=2, num_threads=z)
    print('copy edge cost:', time() - tic)
    tic = time()
    recon_res = recon_res[:, :, :, cut_off:-cut_off, cut_off:-cut_off]
    panorama = recon_res.transpose(2, 0, 3, 1, 4)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>add the threading new array to speedup the following reshape operation<<<<<<<<<<<<<<<<<<<<<<<<<<<
    panorama = threading_copy(panorama, num_threads=z)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>add the threading new array to speedup the following reshape operation<<<<<<<<<<<<<<<<<<<<<<<<<<<
    panorama = panorama.reshape(recon_res.shape[-3], num_y * recon_res.shape[-2], num_x * recon_res.shape[-1])#[D,5*h,7*w]
    print('transpose and reshape cost:', time() - tic)

    ##place the calculated edge into the ori recon_res
    tic = time()
    panorama = place_edge_v4(panorama, left_edge, right_edge, up_edge, down_edge, num_y, num_x, h, w, overlap)
    print('place cost: {:.6f}'.format(time() - tic))

    return panorama

def place_edge_v4(panorama, left_edge, right_edge, up_edge, down_edge, num_y, num_x, h, w, overlap):

    cut_off = overlap // 2
    # tic = time()
    ##place the col
    panorama = parallel_update_panorama_col(panorama, right_edge, left_edge, num_y, num_x, h, w, overlap, cut_off)
    # print('place col cost: {:.6f}'.format(time() - tic))
    # tic = time()
    ##place the row
    panorama = parallel_update_panorama_row(panorama, down_edge, up_edge, num_y, num_x, h, w, overlap, cut_off)
    # print('place row cost: {:.6f}'.format(time() -tic))
    # tic = time()
    ##place the cross
    h_start, h_end = h - overlap - cut_off, h - overlap - cut_off + overlap
    for row in range(num_y-1):
        w_start, w_end = w - overlap - cut_off, w - overlap - cut_off + overlap
        for col in range(num_x - 1):
            panorama[:, h_start:h_end, w_start:w_end] = right_edge[row,col,:, -overlap:, :]+\
                                                        left_edge[row,col+1,:,-overlap:, :]+\
                                                        right_edge[row+1,col,:,:overlap, :]+\
                                                        left_edge[row+1,col+1,:,:overlap, :]
            w_start = w_start + w - overlap
            w_end = w_start+overlap
        h_start = h_start + h - overlap
        h_end = h_start + overlap
    # print('place cross cost: {:.6f}'.format(time() -tic))

    return panorama
def update_panorama_row(panorama, down_edge, up_edge, row, col, h_start, h_end, w_start, w_end, edge_w_start, edge_w_end):
    panorama[:, h_start:h_end, w_start:w_end] = down_edge[row, col, :, :, edge_w_start:edge_w_end] + \
                                                up_edge[row + 1, col, :, :, edge_w_start:edge_w_end]
def parallel_update_panorama_row(panorama, down_edge, up_edge, num_y, num_x, h, w, overlap, cut_off):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        h_start, h_end = h - overlap - cut_off, h - overlap - cut_off + overlap
        for row in range(num_y - 1):
            w_start, w_end = 0, w - overlap - cut_off  # 每一行的第一列的起止放置坐标
            for col in range(num_x):
                if col == 0:
                    edge_w_start, edge_w_end = cut_off, w - overlap
                elif col == num_x - 1:
                    edge_w_start, edge_w_end = overlap, w - cut_off
                else:
                    edge_w_start, edge_w_end = overlap, w - overlap
                futures.append(
                    executor.submit(
                        update_panorama_row,panorama, down_edge, up_edge, row, col, h_start, h_end, w_start, w_end, edge_w_start, edge_w_end
                    )
                )
                w_start = w_end + overlap
                w_end = w_start + w - (1 + (col < (num_x - 2))) * overlap
            h_start = h_start + h - overlap
            h_end = h_start + overlap

        # Ensure all threads are completed
        for future in futures:
            future.result()
    return panorama
def update_panorama_col(panorama, right_edge, left_edge, row, col, h_start, h_end, w_start, w_end, edge_h_start, edge_h_end):
    panorama[:, h_start:h_end, w_start:w_end] = right_edge[row, col, :, edge_h_start:edge_h_end, :] + \
                                                left_edge[row, col + 1, :, edge_h_start:edge_h_end, :]
def parallel_update_panorama_col(panorama, right_edge, left_edge, num_y, num_x, h, w, overlap, cut_off):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        h_start, h_end = 0, h - overlap - cut_off
        for row in range(num_y):
            if row == 0:
                edge_h_start, edge_h_end = cut_off, h - overlap
            elif row == num_y - 1:
                edge_h_start, edge_h_end = overlap, h - cut_off
            else:
                edge_h_start, edge_h_end = overlap, h - overlap
            w_start, w_end = w - overlap - cut_off, w - overlap - cut_off + overlap
            for col in range(num_x - 1):
                futures.append(
                    executor.submit(
                        update_panorama_col,panorama, right_edge, left_edge, row, col, h_start, h_end, w_start, w_end, edge_h_start, edge_h_end
                    )
                )
                w_start = w_start + w - overlap
                w_end = w_start + overlap
            h_start = h_end + overlap
            h_end = h_start + h - (1 + (row < (num_y - 2))) * overlap

        # Ensure all threads are completed
        for future in futures:
            future.result()
    return panorama

def recon_post_process(volume, ms_grid=None, target_z=81):
    if volume.shape[0] > target_z:
        crop = (volume.shape[0] - target_z) // 2
        volume = volume[crop:-crop]
        assert volume.shape[0] == target_z
    if ms_grid is not None:
        pass
    volume = volume.clamp(0, 2 ** 16 - 1)
    volume = (volume - 32768).to(torch.int16) + 32768
    volume = volume.detach().cpu().numpy().view(dtype=np.uint16)
    return volume

def insert_volume(index_volume_queue, volume_stack):
    while not index_volume_queue.empty():
        index, volume = index_volume_queue.get()
        volume_stack[index] = volume
        index_volume_queue.task_done()

def add_save_task(buffer, save_path, img, mip=False):
    buffer.put([save_path, img, mip])

def save_to_file(save_path, img):
    t = time()
    tifffile.imwrite(save_path, img, maxworkers=1)
    log_out('Save %s takes: %.5f s' % (save_path, time() - t))

def saving_task(buffer, exit_flag, name):
    while not (exit_flag.is_set() and buffer.empty()):
        try:
            item = buffer.get(timeout=2)
            # print(f"Consumer {name} buffer size {buffer.qsize()}")
            save_path = item[0]
            img = item[1]
            mip = item[2]
            if mip:
                img = np.max(img, axis=0)[None,]
            save_to_file(save_path, img)
            del img, save_path
            gc.collect()
            buffer.task_done()
        except:
            sleep(2)
            pass
        # time.sleep(2)

def set_finish(consumer_futures, exit_flag):
    exit_flag.set()
    concurrent.futures.wait(consumer_futures)

def insert_volume(index_volume_queue, volume_stack):
    while not index_volume_queue.empty():
        index, volume = index_volume_queue.get()
        volume_stack[index] = volume
        index_volume_queue.task_done()
        
def threading_copy(input, num_threads:int=63, axis=0):
    '''
    :param input: 待复制的input
    :param num_threads: 线程数
    :return: 复制后的array
    '''
    if axis and type(input)==np.ndarray:
        input = input.swapaxes(axis, 0)
    if type(input) ==  np.ndarray:
        shape = input.shape
        dtype = input.dtype
    else:
        shape = (len(input),*input[0].shape)
        dtype = input[0].dtype
    volume_stack_thread = np.empty(shape=shape, dtype=dtype)
    index_volume_queue = queue.Queue(maxsize=num_threads)
    for ind, vol in enumerate(input):
        index_volume_queue.put([ind, vol])

    # 创建线程池
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=insert_volume, args=(index_volume_queue, volume_stack_thread))
        t.start()
        threads.append(t)

    # 等待所有线程完成
    index_volume_queue.join()

    # 停止线程
    for _ in range(num_threads):
        index_volume_queue.put(None)
    for t in threads:
        t.join()

    if axis and type(input)==np.ndarray:
        volume_stack_thread = volume_stack_thread.swapaxes(axis, 0)
    return volume_stack_thread

def get_recon_path(save_folder, proj_name, site, channel, block, frame):
    if not os.path.exists(save_folder): os.makedirs(save_folder)
    recon_name = '%s_S%d_C%d_B%d_T%d.tif' % (proj_name, site ,channel, block, frame)
    recon_path = save_folder + '/' + recon_name
    return  recon_path


def get_block_undistort_path(save_folder, proj_name, site, channel, block, frame):
    save_folder = os.path.dirname(save_folder) + '/block_undistort'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    recon_name = '%s_S%d_C%d_B%d_T%d_block_undistort.tif' % (proj_name, site, channel, block, frame)
    recon_path = save_folder + '/' + recon_name
    return  recon_path

def get_realign_reshape_path(save_folder, proj_name, site, channel, block, frame):
    save_folder = os.path.dirname(save_folder) + '/realign'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    recon_name = '%s_S%d_C%d_B%d_T%d_realign_reshape.tif' % (proj_name, site, channel, block, frame)
    recon_path = save_folder + '/' + recon_name
    return  recon_path


def get_realign_merge_path(save_folder, proj_name, site, channel, block, frame):
    save_folder = save_folder + f'/realign_C{channel}'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    recon_name = '%s_S%d_C%d_B%d_T%d_realign_merge.tif' % (proj_name, site, channel, block, frame)
    recon_path = save_folder + '/' + recon_name
    return  recon_path


def get_vsr_path(save_folder, proj_name, site, channel, block, frame):
    save_folder = os.path.dirname(save_folder) + '/VSR'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    recon_name = '%s_S%d_C%d_B%d_T%d_VSR.tif' % (proj_name, site, channel, block, frame)
    recon_path = save_folder + '/' + recon_name
    return  recon_path


def get_denoise_path(save_folder, proj_name, site, channel, block, frame):
    save_folder = save_folder + '/denoise'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    recon_name = '%s_S%d_C%d_B%d_T%d_denoise.tif' % (proj_name, site, channel, block, frame)
    recon_path = save_folder + '/' + recon_name
    return  recon_path


def log_out(out):
    if not os.path.exists('./log'): os.makedirs('./log')
    with open(f"./log/python_recon_log_{datetime.now().strftime('%Y-%m-%d')}.txt", "a") as file:
        file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   " + out + "\n")
    print(str(out))
    sys.stdout.flush()


def get_psf_mat(psf_path, depth=73):
    psf_path += '.RL.mat'
    if os.path.exists(psf_path):
        log_out('Waiting for loading psf from %s.......' % psf_path)
        t = time()
        psf = np.array(h5py.File(psf_path, 'r').get('psf_4d'), dtype=np.float32)
        if psf.shape[0] == 81:
            psf = torch.tensor(psf[::-1, :, :, :].copy()).permute(0, 1, 3, 2)   # H W D C matlab -> C D W H h5
        elif psf.shape[-1] == 81:
            psf = torch.tensor(psf[:,:,:,::-1].copy()).permute(3, 2, 1, 0)   # C D H W matlab -> W H D C h5
        else:
            raise NotImplementedError
        d = psf.shape[1]
        if d > depth:
            d0 = (d - depth) // 2
            d1 = d0 + depth
            psf = psf[:, d0: d1, :]
        weight = np.array(h5py.File(psf_path, 'r').get('weight'), dtype=np.float32).transpose()[0]
        weight = torch.tensor(weight[::-1].copy())
        weight = weight / weight.max() * 0.8
        print('Loading psf finished, shape:', psf.shape, end=", ")
        print('takes: %f s' % (time() - t))
        return {"psf": psf,
                "weight": weight}
    else:
        raise Exception("psf: %s not exists" % psf_path)


def get_psf_shift_pt(psf_path, scale=15/3, depth=65, offset=0):
    psf_path += '.AI.mat'
    if os.path.exists(psf_path):
        psf = torch.tensor(np.array(h5py.File(psf_path, 'r').get('shift'), dtype=np.float32))
        if psf.shape[0] == 2:
            psf = psf.permute(2,1,0).contiguous()
        d = psf.shape[1]
        if d > depth:
            d0 = (d - depth) // 2 + offset
            d1 = d0 + depth
            assert d0 >= 0, f"psf lowerbound should <0, but d0 is {d0}"
            assert d1 <= d, f"psf upperbound should not exceed psf depth {d}, but d1 is {d1}"
            print(d0, d1)
            psf = psf[:, d0: d1, :]
        psf /= scale
        return psf
    else:
        raise Exception("psf: %s not exists" % psf_path)

def get_input_views(mode):
    input_views = [112, 113, 128, 127, 126, 111, 96, 97, 98, 99, 114, 129, 144, 143, 142, 141, 140, 125, 110,
                   95, 80, 81, 82, 83, 84, 85, 100, 115, 130, 145, 160, 159, 158, 157, 156, 155, 154, 139, 124,
                   109, 94, 79, 64, 65, 66, 67, 68, 69, 70, 71, 86, 101, 116, 131, 146, 161, 175, 174, 173, 172,
                   171, 170, 169, 153, 138, 123, 108, 93, 78, 63, 49, 50, 51, 52, 53, 54, 55, 117, 187, 107, 37]
    if mode == "RL":
        input_views = input_views[::-1]

    return torch.tensor(input_views)

def get_lf_list_2(in_folder, proj_name, site, channel, block, batch_id, wigner_id0, n_wigner=180, sync=False):
    img_id0, img_id1 = (batch_id[0] + wigner_id0) // n_wigner, (batch_id[1] + wigner_id0 - 1) // n_wigner
    if img_id0 == img_id1:
        src_img_path = in_folder + '/C%d/%s_S%d_C%d_B%d_%d.tiff' % (channel, proj_name, site, channel, block, img_id0) # (T, h, w)
        if not os.path.exists(src_img_path):
            raise Exception('File path: %s not exist' % src_img_path)
        else:
            t0 = time()
            src_img = tifffile.imread(src_img_path, key=slice((batch_id[0] + wigner_id0)%n_wigner, (batch_id[1] + wigner_id0 - 1)%n_wigner + 1)).astype(np.float32)
            log_out('Loading file: %s takes: %.5f s' % (src_img_path, time() - t0))
            if sync:
                return block, src_img
            else:
                return src_img
    else:
        src_img_path_0 = in_folder + '/C%d/%s_S%d_C%d_B%d_%d.tiff' % (channel, proj_name, site, channel, block, img_id0) # (T, h, w)
        src_img_path_1 = in_folder + '/C%d/%s_S%d_C%d_B%d_%d.tiff' % (channel, proj_name, site, channel, block, img_id1) # (T, h, w)
        if not os.path.exists(src_img_path_0) or not os.path.exists(src_img_path_1):
            raise Exception('File path: %s not exist \n %s' % (src_img_path_0, src_img_path_1))
        else:
            t0 = time()
            src_img_0 = tifffile.imread(src_img_path_0,key=slice((batch_id[0] + wigner_id0)%n_wigner,n_wigner)).astype(np.float32)
            log_out('Loading file: %s takes: %.5f s' % (src_img_path_0, time() - t0))
            t0 = time()
            src_img_1 = tifffile.imread(src_img_path_1,key=slice(0,(batch_id[1] + wigner_id0)%n_wigner)).astype(np.float32)
            log_out('Loading file: %s takes: %.5f s' % (src_img_path_1, time() - t0))
            all_src_img = np.concatenate((src_img_0, src_img_1), axis=0)
            if sync:
                return block, all_src_img
            return all_src_img

def get_lf_list_full_grid(in_folder, proj_name, site, channel, batch_id, wigner_id0):
    img_path = in_folder + '/C%d/%s_S%d_C%d_%d.tiff' % (channel, proj_name, site, channel, wigner_id0+batch_id[0])
    test_img = tifffile.imread(img_path)
    if len(test_img.shape)==3:
        all_src_img = torch.empty((batch_id[1]-batch_id[0], 81, 683, 945), dtype=torch.float32)
        mode = 1
    if len(test_img.shape)==2:
        all_src_img = torch.empty((batch_id[1]-batch_id[0], 10748, 14304), dtype=torch.float32)
        mode = 0

    for img_idx in range(wigner_id0+batch_id[0], wigner_id0+batch_id[1]):
        img_path = in_folder + '/C%d/%s_S%d_C%d_%d.tiff' % (channel, proj_name, site, channel, img_idx)
        if not os.path.exists(img_path):
            raise Exception('File path: %s not exist' % img_path)
        else:
            t0 = time()
            all_src_img[img_idx-wigner_id0-batch_id[0]] = torch.from_numpy(tifffile.imread(img_path).astype(np.float32))
            log_out('Loading file: %s takes: %.5f s' % (img_path, time() - t0))
    return all_src_img, mode

def get_lf_ind_list(batch_id_list, group_mode, total_wigner_num, nshift=3):
    if nshift == 3: batch_num = 45 
    elif nshift == 5: batch_num = 100
    elif nshift == 15: batch_num = 225
    else: raise Exception('nshift must be 3, 5 or 15')
    batch_step = batch_num if group_mode == 0 else batch_num - nshift**2 + 1
    batch_max_len = batch_num if group_mode == 0 else 2 * batch_num - nshift**2 + 1
    lf_id0 = 0
    while lf_id0 + batch_max_len <= total_wigner_num:
        lf_id1 = lf_id0 + batch_num
        batch_id_list.append([lf_id0, lf_id1])
        lf_id0 += batch_step
    else:
        batch_id_list.append([lf_id0, total_wigner_num])

    return batch_id_list#返回每一个batch的起止wigner索引

def get_scan_config(path):
     if os.path.exists(path):
        scan_config = np.array(Image.open(path))
        return scan_config
     else:
         raise Exception('The path of scan_config : %s does not exits' % path)
