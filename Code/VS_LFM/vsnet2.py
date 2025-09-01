import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.transforms as transforms
# import math

class Net(nn.Module):
    def __init__(self, angRes, factor):
        super(Net, self).__init__()
        channels = 64
        self.channels = channels
        self.angRes = angRes
        self.factor = factor
        layer_num = 16


        self.MHSA_params = {}
        self.MHSA_params['num_heads'] = 8
        self.MHSA_params['dropout'] = 0.

        ##################### Initial Convolution #####################
        self.conv_init0 = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
        )
        self.conv_init = nn.Sequential(
              CascadedBlocks(layer_num, channels, angRes),
          )
        ####################### UP Sampling ###########################
        self.upsampling = nn.Sequential(
            nn.Conv2d(channels, channels*self.factor ** 2, kernel_size=1, padding=0, dilation=1, bias=False),
            # nn.Conv3d(channels, channels*self.factor ** 2, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.PixelShuffle(self.factor),
            nn.LeakyReLU(0., inplace=True),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.Conv3d(channels, 1, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
        )

    def forward(self, lr):
        # Bicubic
        b,n,h,w = lr.shape

        H,W = h*self.factor, w*self.factor

        lr = lr.reshape(b, 1, n, h, w) #b,1,n,h,w

        # Bicubic
        lr_upscale = F.interpolate(lr.reshape(b* lr.shape[1]* n, 1, h, w),scale_factor= self.factor, mode='bicubic', align_corners=False).reshape(b,1,n,H,W)

        # Initial Convolution
        buffer_init = self.conv_init0(lr)
        buffer = self.conv_init(buffer_init)+buffer_init

        buffer = buffer.permute(0, 2, 1, 3, 4).reshape(b*n, self.channels, h, w)

        buffer = self.upsampling(buffer).reshape(b,n, 1, H, W).permute(0, 2, 1, 3, 4)
        out = buffer + lr_upscale

        out = out.reshape(b,1,n,H,W).squeeze(1)
        return out

class Vsnet_light(nn.Module):
    def __init__(self, ch, angRes):
        super(Vsnet_light, self).__init__()
                
        # self.relu = nn.ReLU(inplace=True)
        S_ch, A_ch = ch, ch
        self.angRes = angRes
        self.spaconv  = SpatialConv(ch)
        self.angconv  = AngularConv(ch, angRes, A_ch)
        self.fuse = nn.Sequential(
                nn.Conv3d(in_channels = S_ch+A_ch*1, out_channels = ch, kernel_size = 1, stride = 1, padding = 0, dilation=1,bias=False),
                nn.LeakyReLU(0., inplace=True),
                nn.Conv3d(ch, ch, kernel_size = (1,3,3), stride = 1, padding = (0,1,1),dilation=1,bias=False))
    
    def forward(self,x):
        # b, n, c, h, w = x.shape
        b, c, n, h, w = x.shape
        s_out = self.spaconv(x)
        a_out = self.angconv(x)
        out = torch.cat((s_out, a_out), 1)

        out = self.fuse(out)

        return out + x# out.contiguous().view(b,n,c,h,w) + x


class SpatialConv(nn.Module):
    def __init__(self, ch):
        super(SpatialConv, self).__init__()
        self.spaconv_s = nn.Sequential(
                    nn.Conv3d(in_channels = ch, out_channels = ch, kernel_size = (1,3,3), stride = (1,1,1), padding = (0,1,1), dilation=(1,1,1),bias=False),
                    nn.LeakyReLU(0., inplace=True),
                    nn.Conv3d(in_channels = ch, out_channels = ch, kernel_size = (1,3,3), stride = (1,1,1), padding = (0,1,1), dilation=(1,1,1),bias=False),
                    nn.LeakyReLU(0., inplace=True)
                    )

    def forward(self,fm):

        return self.spaconv_s(fm) 



class AngularConv(nn.Module):
    def __init__(self, ch, angNum, AngChannel):
        super(AngularConv, self).__init__()
        self.angconv = nn.Sequential(
            nn.Conv3d(ch*angNum, AngChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0., inplace=True),
            nn.Conv3d(AngChannel, AngChannel * angNum, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0., inplace=True),
            # nn.PixelShuffle(angRes)
        )
        # self.an = angRes

    def forward(self,fm):
        b, c, n, h, w = fm.shape
        a_in = fm.contiguous().view(b,c*n,1,h,w)
        out = self.angconv(a_in).view(b,-1,n,h,w) # n == angRes * angRes
        return out
    


class CascadedBlocks(nn.Module):
    '''
    Hierarchical feature fusion
    '''
    def __init__(self, n_blocks, channel, angRes):
        super(CascadedBlocks, self).__init__()
        self.n_blocks = n_blocks
        body = []
        for i in range(n_blocks):
            body.append(Vsnet_light(channel, angRes))
        self.body = nn.Sequential(*body)
        self.conv = nn.Conv3d(channel, channel, kernel_size = (1,3,3), stride = (1,1,1), padding = (0,1,1), dilation=1,bias=False)

    def forward(self, x):
        # x = x.permute(0, 2, 1, 3, 4)
        # b, n, c, h, w = x.shape     
        buffer = x
        for i in range(self.n_blocks):
            buffer = self.body[i](buffer)        
        # buffer = self.conv(buffer.contiguous().view(b*n, c, h, w))
        buffer = self.conv(buffer) + x
        # buffer = buffer.contiguous().view(b,n, c, h, w) + x
        return buffer# buffer.permute(0, 2, 1, 3, 4)

class get_loss(nn.Module):
    def __init__(self, args):
        super(get_loss, self).__init__()
        self.criterion_Loss = torch.nn.L1Loss()

    def forward(self, SR, HR):
        loss = self.criterion_Loss(SR, HR)

        return loss
