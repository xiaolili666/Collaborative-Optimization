from typing import Callable

import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch.utils.hooks import RemovableHandle
from P1.MambaRSDD.VMamba.classification.models.vmamba import vmamba_tiny_s1l8
from backbone.Shunted_Transformer.SSA import shunted_b
from einops import rearrange
# ===============================================================================================================

TRAIN_SIZE = 256

class SalHead(nn.Module):
    def __init__(self, in_channel):
        super(SalHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channel, 1, 1, stride=1, padding=0),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)



class MAG(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb = vmamba_tiny_s1l8()
        for name, parameter in self.rgb.named_parameters():
            if 'fitune' in name:
                parameter.requires_grad = True
            # if 'mlp' in name:
            #     pass
            else:
                parameter.requires_grad = False

        self.dep = vmamba_tiny_s1l8()
        for name, parameter in self.dep.named_parameters():
            if 'fitune' in name:
                parameter.requires_grad = True
            # if 'mlp' in name:
            #     pass
            else:
                parameter.requires_grad = False




        self.decoder_f = Decoer()
        self.predtrans = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=768, kernel_size=3, stride=1, padding=1, groups=768),
            nn.BatchNorm2d(768),
            nn.GELU(),
            nn.Conv2d(in_channels=768, out_channels=1, kernel_size=1, stride=1)
        )
        #融合2
        self.fr4 = FuseModule(768,768)
        self.fr3 = FuseModule(768, 384)
        self.fr2 = FuseModule(384, 192)
        self.fr1 = FuseModule(192, 96)

        self.fd4 = FuseModule(768, 768)
        self.fd3 = FuseModule(768, 384)
        self.fd2 = FuseModule(384, 192)
        self.fd1 = FuseModule(192, 96)

        self.conv1 = nn.Conv2d(in_channels=768, out_channels=512, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=1)



        self.SA96 = SalHead(96)
        self.SA192 = SalHead(192)
        self.SA384 = SalHead(384)
        self.SA768 = SalHead(768)

    def forward(self, x_rgb, x_d):
        rgb1, rgb2, rgb3, rgb4, rgb5 = self.rgb(x_rgb)
        x_d = torch.cat([x_d,x_d,x_d],dim=1)
        d1, d2, d3, d4, d5 = self.dep(x_d)

        #融合

        add4r = self.fr4(rgb5, rgb4)
        add3r = self.fr3(rgb4, rgb3)
        add2r = self.fr2(rgb3, rgb2)
        add1r = self.fr1(rgb2, rgb1)

        add4d = self.fd4(d5, d4)
        add3d = self.fd3(d4, d3)
        add2d = self.fd2(d3, d2)
        add1d = self.fd1(d2, d1)

        add4 = add4d + add4r
        add3 = add3d + add3r
        add2 = add2d + add2r
        add1 = add1d + add1r



        # print("add4rf", add4.shape)
        # print("add3rf", add3.shape)
        # print("add2rf", add2.shape)
        # print("add1rf", add1.shape)
        # print("add5f", add5f.shape)

        #解码
        pre1, pre2, pre3, pre4 = self.decoder_f(add1, add2, add3, add4)
        pre1 = F.interpolate(self.SA96(pre1), 256, mode="bilinear", align_corners=True)
        pre2 = F.interpolate(self.SA192(pre2), 256, mode="bilinear", align_corners=True)
        pre3 = F.interpolate(self.SA384(pre3), 256, mode="bilinear", align_corners=True)
        pre4 = F.interpolate(self.SA768(pre4), 256, mode="bilinear", align_corners=True)
        # add1 = F.interpolate(self.SA96(add1), 256, mode="bilinear", align_corners=True)
        # add2 = F.interpolate(self.SA192(add2), 256, mode="bilinear", align_corners=True)
        # add3 = F.interpolate(self.SA384(add3), 256, mode="bilinear", align_corners=True)
        # add4 = F.interpolate(self.SA768(add4), 256, mode="bilinear", align_corners=True)
        # f1 = F.interpolate(self.SA768(rgb5+d5), 256, mode="bilinear", align_corners=True)
#out1 torch.Size([2, 96, 64, 64])
# out1 torch.Size([2, 96, 64, 64])
# out1 torch.Size([2, 96, 64, 64])

        return pre1, pre2, pre3, pre4,self.conv4(add1),self.conv3(add2),self.conv2(add3),self.conv1(add4)

#融合模块
class FuseModule(nn.Module):
    def __init__(self,channel1, channel2):
        super(FuseModule, self).__init__()
        self.drop = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Sequential(nn.Conv2d(channel1, channel2, 1),
                                  nn.BatchNorm2d(channel2),
                                  nn.ReLU())
    def forward(self, x1, x2):
        #
        x1 = self.conv(x1)
        # print("x12", x1.shape)
        b1, c1, h1, w1 = x1.shape
        b2, c2, h2, w2 = x2.shape
        x1 = F.interpolate(input=x1, scale_factor=h2//h1, mode='bilinear', align_corners=True)

        x_q = rearrange(x1, 'b c h w -> b c (h w)')
        x_k = rearrange(x1, 'b c h w -> b (h w) c')
        attention1_map = F.sigmoid(torch.bmm(x_q, x_k))
        attention1_map = torch.mean(attention1_map, dim=2)
        attention1_map = self.drop(rearrange(attention1_map, 'b c -> b c 1 1'))

        x2_q = rearrange(x2, 'b c h w -> b c (h w)')
        x2_k = rearrange(x2, 'b c h w -> b (h w) c')
        attention2_map = F.sigmoid(torch.bmm(x2_q, x2_k))
        attention2_map = torch.mean(attention2_map, dim=2)
        attention2_map = self.drop(rearrange(attention2_map, 'b c -> b c 1 1'))
        add_max = attention1_map + attention2_map

        add_max = self.sigmoid(add_max)
        # print("add_max", add_max.shape)
        x1_out = x1 * add_max
        x2_out = x2 * add_max
        # print("x1_out",x1_out.shape)
        # print("x2_out", x2_out.shape)
        mul_f = x1_out * x2_out
        out = mul_f + x1_out + x2_out

        return out



#de
class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Sequential(nn.Conv2d(in_channel, depth, 1, 1),
                                           nn.BatchNorm2d(depth),
                                           nn.ReLU())
        self.atrous_block6 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6),
                                           nn.BatchNorm2d(depth),
                                           nn.ReLU())
        self.atrous_block12 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12),
                                            # nn.Conv2d(in_channel, depth, kernel_size=(1, 3), padding=(0, 12), dilation=(1, 12)),
                                            # nn.Conv2d(depth, depth, kernel_size=(3, 1), padding=(12, 0), dilation=(12, 1)),
                                            nn.BatchNorm2d(depth),
                                            nn.ReLU())
        self.atrous_block18 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18),
                                            # nn.Conv2d(in_channel, depth, kernel_size=(1, 3), padding=(0, 18), dilation=(1, 18)),
                                            # nn.Conv2d(depth, depth, kernel_size=(3, 1), padding=(18, 0), dilation=(18, 1)),
                                            nn.BatchNorm2d(depth),
                                            nn.ReLU())
        self.conv_1x1_output = nn.Sequential(nn.Conv2d(depth * 4, 4*depth, 1),
                                             nn.BatchNorm2d(4*depth),
                                             nn.ReLU())

    def forward(self, x):

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        net = net + x

        return net


class Decoer(nn.Module):
    def __init__(self):
        super().__init__()

        # self.aspp5 = ASPP(in_channel=768, depth=192)
        self.aspp4 = ASPP(in_channel=768, depth=192)
        self.aspp3 = ASPP(in_channel=384, depth=96)
        self.aspp2 = ASPP(in_channel=192, depth=48)
        self.aspp1 = ASPP(in_channel=96, depth=24)


        self.up2 = nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear')
        self.conv4 = nn.Sequential(nn.Conv2d(1536, 768, 1),
                                   nn.BatchNorm2d(768),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear'))
        self.conv3 = nn.Sequential(nn.Conv2d(1152, 384, 1),
                                   nn.BatchNorm2d(384),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear'))
        self.conv2 = nn.Sequential(nn.Conv2d(576, 192, 1),
                                   nn.BatchNorm2d(192),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear'))
        self.conv1 = nn.Sequential(nn.Conv2d(288, 96, 1),
                                   nn.BatchNorm2d(96),
                                   nn.ReLU(),
                                   nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear'))

    def forward(self, f1, f2, f3, f4):

        cat4 = self.aspp4(self.up2(f4))
        cat3 = torch.cat((f3, cat4), dim=1)
        cat3 = self.aspp3(self.conv3(cat3))
        cat2 = torch.cat((f2, cat3), dim=1)
        cat2 = self.aspp2(self.conv2(cat2))
        cat1 = torch.cat((cat2, f1), dim=1)
        cat1 = self.aspp1(self.conv1(cat1))

        return cat1,cat2,cat3,cat4

#

# if __name__ == '__main__':
#
#     input_rgb = torch.randn(2, 3, 256, 256).cuda()
#     input_d = torch.randn(2, 1, 256, 256).cuda()
#     net = MAG().cuda()
#     out = net(input_rgb, input_d)
#     for out in out:
#         print("out1", out.shape)



if __name__ == '__main__':
    a = torch.randn(2, 3, 256, 256).cuda()
    b = torch.randn(2, 1, 256, 256).cuda()
    model = MAG().cuda()
    #
    out = model(a, b)
    # from thop import profile
    #
    # flops, params = profile(model, (a, b))
    # print("flops:%.2f G，params:%.2f M" % (flops / 1e9, params / 1e6))
    from FLOP import CalParams
    # flops, params = profile(model, inputs=(a,b))
    # flops, params = clever_format([flops, params], "%.2f")
    #  22.27M 15.20G
    CalParams(model,a,b)
    print('Total params %.2f'%(sum(p.numel() for p in model.parameters()) / 1000000.0))