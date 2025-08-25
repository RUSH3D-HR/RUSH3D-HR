import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import register

class upsample3dhw(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale=scale

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=(1,self.scale,self.scale),align_corners=False,mode='trilinear')
        return x
    
def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
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
    return ret

def rolldim_resample(input, shift, crop):
    """ shift correspondingly and concat
    input: [C,H,W]
    shift: [C,D,2]
    return:[C,D,H,W]
    """
    grid = make_coord([i for i in [input.shape[-2], input.shape[-1]]], flatten=False).flip(-1).unsqueeze(0).to(input.device) # 1,H,W,2
    grid = grid[:,crop:-crop,crop:-crop,:]
    inp_all = []
    input = input.unsqueeze(1) # V,1,H,W
    for i in range(shift.shape[1]):
        new_grid = grid + shift[:,i].flip(-1).unsqueeze(1).unsqueeze(1) * 2 / torch.tensor([input.shape[-1], input.shape[-2]], device=input.device) # V, H,W,2
        inp_all.append(F.grid_sample(input, new_grid, mode='bilinear', align_corners=False))
    return torch.cat(inp_all, dim=1)


@register('serenet')
class SERENET(nn.Module):
    def __init__(self, inChannels, negative_slope=0.1, usingbias=False, reset_param=False):
        super().__init__()
        self.fusion =  nn.Sequential( # 8conv
            nn.Conv3d(inChannels, 64, kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias ),nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv3d(64, 32, kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias ),nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv3d(32, 32, kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias ),nn.LeakyReLU(negative_slope, inplace=True),

            nn.Conv3d(32, 16, kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias ),nn.LeakyReLU(negative_slope, inplace=True),
            upsample3dhw(2),
            nn.Conv3d(16,16,kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),

            nn.Conv3d(16,8,kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),
            upsample3dhw(2.5),
            nn.Conv3d(8,8,kernel_size=(3,3,3),stride=1, padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),
            
            # OLD
            nn.Conv3d(8,4,kernel_size=(3,3,3),stride=1, padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv3d(4,1,kernel_size=(3,3,3),stride=1, padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),
            # END OLD
            # nn.Conv3d(8,4,kernel_size=(3,3,3),stride=1,padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),
            # upsample3dhw(1.25),
            # nn.Conv3d(4,4,kernel_size=(3,3,3),stride=1, padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),

            # nn.Conv3d(4,1,kernel_size=(3,3,3),stride=1, padding=(1,1,1), bias=usingbias),nn.LeakyReLU(negative_slope, inplace=True),
        )

        if reset_param: self.reset_params()
        
    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal(0, 0.01)
                m.bias.data.zero_()


    def forward(self, inp, psf, transition, scale=1., multi_block=False):
        inp = rolldim_resample(inp, psf, 27).unsqueeze(0) # 1,81,101,477,477->1,81,101,423,423
        # if multi_block:
        if True:
            with torch.no_grad():
                block_inp = [inp[:,:,:,:inp.shape[-2]//2+6,:inp.shape[-1]//2+6],
                            inp[:,:,:,:inp.shape[-2]//2+6,inp.shape[-1]//2-6:],
                            inp[:,:,:,inp.shape[-2]//2-6:,:inp.shape[-1]//2+6], 
                            inp[:,:,:,inp.shape[-2]//2-6:,inp.shape[-1]//2-6:]
                            ] 
                block_ret = []
                for block in block_inp:
                    block_ret.append(self.fusion(block))
            # ret = torch.cat([torch.cat([block_ret[0][:,:,:,:-24,:-24],block_ret[1][:,:,:,:-24,24:]],dim=-1),torch.cat([block_ret[2][:,:,:,24:,:-24],block_ret[3][:,:,:,24:,24:]],dim=-1)], dim=-2)
            ret = torch.cat([torch.cat([block_ret[0][:,:,:,:-30,:-30],block_ret[1][:,:,:,:-30,30:]],dim=-1),torch.cat([block_ret[2][:,:,:,30:,:-30],block_ret[3][:,:,:,30:,30:]],dim=-1)], dim=-2)
            del block_inp, block_ret
        else:
            with torch.no_grad():
                ret = self.fusion(inp)
            
        ret = F.interpolate(ret, scale_factor = (1, scale, scale),mode='trilinear',align_corners=False).squeeze(0).squeeze(0)
        # ret *= 10
        # ret /= 10
        # add the sigmoid calculation
        ret = ret[:,50:-50,50:-50]
        ret[:,:,:50] = ret[:,:,:50]*transition.reshape(1,1,*transition.shape)#left
        ret[:,:,-50:] = ret[:,:,-50:]*(1 - transition).reshape(1,1,*transition.shape)#right
        ret[:,:50,:] = ret[:,:50,:]*transition.reshape(1,*transition.shape,1)#up
        ret[:,-50:,:] = ret[:,-50:,:]*(1-transition).reshape(1,*transition.shape,1)#down
        return ret 