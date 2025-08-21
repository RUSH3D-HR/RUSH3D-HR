import torch
import torch.nn as nn
from .models import register
import torch.nn.functional as F

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


def rolldim_resample(input, shift):
    """ shift correspondingly and concat
    input: [C,H,W]
    shift: [C,D,2]
    return:[C,D,H,W]
    """
    grid = make_coord([i for i in [input.shape[-2], input.shape[-1]]], flatten=False).flip(-1).unsqueeze(0).to(input.device) # 1,H,W,2
    new_grid = grid + shift.view(-1, 2).flip(-1).unsqueeze(1).unsqueeze(1) * 2 / torch.tensor([input.shape[-1], input.shape[-2]], device=input.device) # C*D, H,W,2
    input = input.unsqueeze(1).repeat(1, shift.shape[1], 1, 1).view(-1, 1, input.shape[-2], input.shape[-1]) # C*D,1,H,W
    inp_all = F.grid_sample(input, new_grid, mode='bilinear', align_corners=False).view(shift.shape[0], shift.shape[1], input.shape[-2], input.shape[-1])

    # del grid, new_grid, input, shift
    # torch.cuda.empty_cache()
    return inp_all


class upsample3dhw(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        x = F.interpolate(x, scale_factor=(1 ,self.scale ,self.scale), align_corners=False, mode='trilinear')

        return x


@register('vdfnetV21')
class vdfnet_V21(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.init_feature = nn.Sequential(
            nn.Conv3d(in_ch, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=3 ,stride=1, padding=1), nn.ReLU(),

            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv3d(32, 16, kernel_size=3 ,stride=1, padding=1), nn.ReLU(),

            upsample3dhw(2),

            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv3d(16, 8, kernel_size=3, stride=1, padding=1), nn.ReLU(),

            upsample3dhw(2),

            nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv3d(8, 4, kernel_size=3, stride=1, padding=1), nn.ReLU(),

            nn.Conv3d(4, 1, kernel_size=3, stride=1, padding=1), nn.ReLU() )


    def forward(self, x, psf, scale=1.25):
        x =  rolldim_resample(x, psf).unsqueeze(0)
        with torch.no_grad():
            feat = self.init_feature(x) # [b, c, d, h, w]
            out = F.interpolate(feat, scale_factor=(1,scale,scale), mode='trilinear', align_corners=False)

        return out
