import os
import numpy as np
import tifffile
from tqdm import trange
import torch
import concurrent.futures as futures
from concurrent.futures import ThreadPoolExecutor


class BlockGenerator:
    def __init__(self, center_pt, block_x=7, block_y=5, step_x=1965, step_y=1965, nx=161, ny=161, n=15):

        self.center_pt = center_pt
        self.block_x = block_x
        self.block_y = block_y
        self.step_x = step_x
        self.step_y = step_y
        self.nx, self.ny = nx, ny
        self.block_h, self.block_w = n * ny, n * nx
        self.center_xs = np.arange(center_pt[1]-step_x*(block_x//2),
                                   center_pt[1]+step_x*(block_x//2)+1, step_x, dtype=int)
        self.center_ys = np.arange(center_pt[0]-step_y*(block_y//2),
                                   center_pt[0]+step_y*(block_y//2)+1, step_y, dtype=int)
    def single_thread_gen_xblocks(self, block_frames, im_stack, y, remove_pad=0):
        ys, ye = self.center_ys[y] - \
            self.block_h // 2, self.center_ys[y] + \
            self.block_h // 2 + 1
        p = remove_pad // 2

        for x in range(self.block_x):
            xs, xe = self.center_xs[x] - \
                self.block_w // 2, self.center_xs[x] + \
                self.block_w // 2 + 1
            for i, im in enumerate(im_stack):
                block_frames[y * self.block_x + x,
                             i] = im[ys + p:ye - p, xs + p:xe - p]
                
    def single_thread_gen_multi_view_xblocks(self, block_frames, im_stack, y, remove_pad=0):
        ys, ye = self.center_ys[y] - \
            self.ny // 2, self.center_ys[y] + \
            self.ny // 2 + 1
        p = remove_pad // 2

        for x in range(self.block_x):
            xs, xe = self.center_xs[x] - \
                self.nx // 2, self.center_xs[x] + \
                self.nx // 2 + 1

            for i, im in enumerate(im_stack):
                block_frames[y * self.block_x + x, i] = im[:, ys + p:ye - p, xs + p:xe - p]

    def gen_blocks(self, im_stack, remove_pad=0):
        block_frames = np.zeros((self.block_x * self.block_y, len(im_stack),
                                self.block_h, self.block_w), dtype=np.float32)   # blocks, frames, nd(h, w)

        with ThreadPoolExecutor(max_workers=self.block_y, thread_name_prefix="gen_block") as executor:
            future_list = []
            for y in trange(self.block_y):
                future_list.append(
                    executor.submit(self.single_thread_gen_xblocks,
                                    block_frames, im_stack, y, remove_pad)
                )
            for f in future_list:
                f.result()
        return block_frames
    
    def gen_blocks_multi_view(self, im_stack, view=81, remove_pad=0):
        block_frames = np.zeros((self.block_x * self.block_y, len(im_stack), view,
                                self.nx, self.ny), dtype=np.float32)   # blocks, frames, view, nd(h, w)

        with ThreadPoolExecutor(max_workers=self.block_y, thread_name_prefix="gen_block") as executor:
            future_list = []
            for y in trange(self.block_y):
                future_list.append(
                    executor.submit(self.single_thread_gen_multi_view_xblocks,
                                    block_frames, im_stack, y, remove_pad)
                )
            for f in future_list:
                f.result()
        return block_frames