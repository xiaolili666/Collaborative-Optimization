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

from backbone.Shunted_Transformer.SSA import shunted_b
# from backbone.PVTv2.pvtv2_encoder import pvt_v2_b1
from einops import rearrange
# from Des.fuse_test import DynamicConv2d
# :0.850; maxEm:0.935; maxFm:0.897; MAE:0.0575.
# /media/pc12/data/LYQ/model/train_test1/Pth2/Net1_12_2025_04_10_15_33_last.pth66
try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import irfft2
    from torch.fft import rfft2
    def rfft(x, d):
        t = rfft2(x, dim = (-d,-1))
        return torch.stack((t.real, t.imag), -1)
    def irfft(x, d, signal_sizes):
        return irfft2(torch.complex(x[:,:,0], x[:,:,1]), s = signal_sizes, dim = (-d,-1))


#Net2_25

TRAIN_SIZE = 256
#domain prompt
class DomainPromptDepFuse(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()


        self.conv = nn.Conv2d(inc * 2 , outc, kernel_size=1, stride=1, padding=0)
        self.region_module = RegionModule(inc, window_size=7)
        self.detail_module = DetailModule(inc)

        self.local_att = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc, outc, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(outc),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inc, outc, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc, outc, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(outc),
        )
        self.duf = DiffusionModule()
    def forward(self, x1, x2):
        # 初始的
             #高频特征
        fft = self.extract_high_freq_features(x1)
        x2 = self.duf(fft,x2)


        r= self.detail_module (x1) # rgb的细节
        d = self.region_module (x2)  #dep的全局

        #相关
        r_out = r * x1 + r
        d_out = x2 * d + d
        out = r_out + d_out

        return out

    def extract_high_freq_features(self, x, alpha=0.3):
        """
        提取高频特征。

        x: (B, C, H, W) 的特征张量/图像张量
        alpha: 低频区域的相对大小，越大表示低频区域越大
        """
        # print(x)
        B, C, H, W = x.shape

        # 1) 先把 x 转到频域
        # 注意：如果 x 是实数张量，结果为复数形式的实数张量 (实部和虚部分开存储)
        X_freq = rfft(x, 2)

        # 2) 将频率坐标中心移到图像中心方便操作
        # 自定义 fftshift 替代实现
        def fftshift(tensor, dim):
            shifted_tensor = tensor
            for d in dim:
                n = tensor.size(d)
                p2 = (n + 1) // 2
                idx = torch.cat((torch.arange(p2, n), torch.arange(0, p2))).to(tensor.device)
                shifted_tensor = shifted_tensor.index_select(d, idx)
            return shifted_tensor

        X_freq_shift = fftshift(X_freq, dim=(-2, -1))

        # 3) 计算要置零的低频区域大小
        low_freq_h = int(H * alpha / 2)
        low_freq_w = int(W * alpha / 2)

        # 4) 对中心低频区域置零
        c_h, c_w = H // 2, W // 2
        X_freq_shift[..., c_h - low_freq_h:c_h + low_freq_h,
        c_w - low_freq_w:c_w + low_freq_w, :] = 0

        # 5) 将频域中心移回
        def ifftshift(tensor, dim):
            shifted_tensor = tensor
            for d in dim:
                n = tensor.size(d)
                p2 = n // 2
                idx = torch.cat((torch.arange(p2, n), torch.arange(0, p2))).to(tensor.device)
                shifted_tensor = shifted_tensor.index_select(d, idx)
            return shifted_tensor

        X_freq_cut = ifftshift(X_freq_shift, dim=(-2, -1))

        # 6) 逆傅里叶变换，得到只保留高频分量的结果
        x_high_freq = irfft(X_freq_cut, 2, signal_sizes=(H, W))

        # 7) 为了后续网络处理或可视化，一般取实部或绝对值
        x_high_freq_real = x_high_freq  # irfft 结果已经是实数

        return x_high_freq_real




#区域感知
class RegionModule(nn.Module):
    def __init__(self, in_channels, window_size):
        super().__init__()
        # Multi-head Self Attention

        self.conv1x1 = nn.Conv2d(in_channels, in_channels, 1)


        # Stripe Convolutions
        self.sconv_1xws = nn.Conv2d(in_channels, in_channels, kernel_size=(1, window_size),
                                    padding=(0, window_size // 2))
        self.sconv_wsx1 = nn.Conv2d(in_channels, in_channels, kernel_size=(window_size, 1),
                                    padding=(window_size // 2, 0))
        # Depth-wise Convolution ws × ws
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=window_size, padding=window_size // 2,groups=in_channels)

        # Feed-Forward Network (FFN)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels,1),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
             )

    def forward(self, x):
        B,C,H,W =x.shape
        # Multi-head Self Attention
       
        x_ln = self.conv1x1(x)

        # Stripe convolutions
        sconv_1xws_out = self.sconv_1xws(x_ln)
        sconv_wsx1_out = self.sconv_wsx1(x_ln)
        sc_out = sconv_1xws_out + sconv_wsx1_out

        # Depth-wise convolution
        dw_out = self.dw_conv(sc_out)

        # Concatenate stripe outputs
        combined = torch.cat([dw_out, x], dim=1)
        out = self.conv1(combined)

        return out




#细节感知
class DetailModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Multiple pooling layers with different kernel sizes
        self.pool_5x5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool_9x9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool_13x13 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        # self.pool_5x5 = nn.AdaptiveAvgPool2d(1)
        # self.pool_9x9 = nn.AdaptiveAvgPool2d(5)
        # self.pool_13x13 = nn.AdaptiveAvgPool2d(9)
        # Convolution layers
        self.conv1x1 = nn.Conv2d(in_channels , in_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels , in_channels, kernel_size=3, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )

        self.conv3 = nn.Conv2d(in_channels * 3 , in_channels, kernel_size=1)

        # self.conv4 = nn.Conv2d(in_channels , in_channels, kernel_size=5)
        # self.conv5 = nn.Conv2d(in_channels , in_channels, kernel_size=9)



    def forward(self, x):
        # Pooling outputs

        res = self.conv1x1(x)
        x = self.conv1(x)

        pool_5 = self.pool_5x5(x)    #pool_5 torch.Size([2, 512, 8, 8])
        pool_9 =  self.pool_9x9(x) # pool_9 torch.Size([2, 512, 8, 8])
        pool_13 = self.pool_13x13(x)

        # Concatenate pooling results
        combined_pool = torch.cat([pool_9, pool_5, pool_13], dim=1)

        # Convolution layers
        x2 = self.conv3(combined_pool)

        # Final layers
        out = x2  + res

        return  out




class MAG1(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.rgb_backbone = shunted_b(pretrained)
        self.d_backbone = shunted_b(pretrained)
        # self.conv =  nn.Conv2d(in_channels=320, out_channels=256, kernel_size=1)
        self.dp1 = DomainPromptDepFuse(64,64)
        self.dp2 = DomainPromptDepFuse(128,128)
        self.dp3 = DomainPromptDepFuse(256,256)
        self.dp4 = DomainPromptDepFuse(512,512)



        self.mcm3 = MCM(inc=512, outc=256)
        self.mcm2 = MCM(inc=256, outc=128)
        self.mcm1 = MCM(inc=128, outc=64)
        # Pred
        self.predtrans = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, groups=512),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1)
        )


    def forward(self, x_rgb, x_d):
        # rgb  rgb1, rgb2 ,rgb3, rgb4   ([2, 76, 64, 64])  ([2, 152, 32, 32])  ([2, 336, 16, 16])  ([2, 672, 8, 8])
        rgb_1, rgb_2, rgb_3, rgb_4 = self.rgb_backbone(x_rgb)



        # d
        x_d = torch.cat([x_d, x_d, x_d], dim=1)
        d_1, d_2, d_3, d_4 = self.d_backbone(x_d)


        # fuse_4  = self.dp4(rgb_4,d_4)
        # fuse_3  = self.dp3(rgb_3,d_3)#self.dp3(self.conv(rgb_3),self.conv(d_3))#
        # fuse_2  = self.dp2(rgb_2,d_2)
        # fuse_1  = self.dp1(rgb_1,d_1)

        fuse_4  = rgb_4+d_4
        fuse_3  = rgb_3+d_3
        fuse_2  = rgb_2+d_2
        fuse_1  = rgb_1+d_1





        # 几何变换应用

        pre4 = F.interpolate(self.predtrans(fuse_4), 256, mode="bilinear", align_corners=True)
        pre3, xf_3 = self.mcm3(fuse_3, fuse_4)
        pre2, xf_2 = self.mcm2(fuse_2, xf_3)
        pre1, xf_1 = self.mcm1(fuse_1, xf_2)



        return pre1, pre2, pre3, pre4,fuse_1,fuse_2,fuse_3,fuse_4



#扩散
class DiffusionModule(nn.Module):
    def __init__(self, alpha=0.5, sigma=1.0):
        super(DiffusionModule, self).__init__()
        self.alpha = alpha  # 融合权重
        self.sigma = sigma  # 高斯核参数

    def forward(self, feature_src, feature_dst):
        B, C, H, W = feature_src.size()

        # 展平特征
        feature_src_flat = feature_src.view(B, C, -1)  # B x C x (H * W)
        feature_dst_flat = feature_dst.view(B, C, -1)

        # 计算相似性矩阵 (高斯核)
        similarity = torch.einsum('bcn,bcm->bnm', feature_src_flat, feature_src_flat)  # B x (H*W) x (H*W)
        similarity = similarity / (C ** 0.5)  # 归一化

        # 高斯核
        diff = similarity - similarity.mean(dim=-1, keepdim=True)
        affinity = torch.exp(-diff ** 2 / (2 * self.sigma ** 2))

        # 归一化扩散矩阵
        diffusion_matrix = F.normalize(affinity, p=1, dim=-1)  # 行归一化

        # 扩散操作
        diffused_feature = torch.einsum('bnm,bcm->bcn', diffusion_matrix, feature_src_flat)
        diffused_feature = diffused_feature.view(B, C, H, W)  # 恢复形状

        # 融合结果
        output_feature = self.alpha * diffused_feature + (1 - self.alpha) * feature_dst
        return output_feature




#  DWPWGELU
class DWPWConv(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, padding=1, stride=1, groups=inc),
            nn.BatchNorm2d(inc),
            nn.GELU(),
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )

    def forward(self, x):

        return self.conv(x)



class MCM(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.rc = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, padding=1, stride=1, groups=inc),
            nn.BatchNorm2d(inc),
            nn.GELU(),
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )
        self.predtrans = nn.Sequential(
            nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, padding=1, groups=outc),
            nn.BatchNorm2d(outc),
            nn.GELU(),
            nn.Conv2d(in_channels=outc, out_channels=1, kernel_size=1)
        )

        self.rc2 = nn.Sequential(
            nn.Conv2d(in_channels=outc * 2, out_channels=outc * 2, kernel_size=3, padding=1, groups=outc * 2),
            nn.BatchNorm2d(outc * 2),
            nn.GELU(),
            nn.Conv2d(in_channels=outc * 2, out_channels=outc, kernel_size=1, stride=1),
            nn.BatchNorm2d(outc),
            nn.GELU()
        )




    def forward(self, x1, x2):

        x2_upsample = self.upsample2(x2)  # 上采样
        x2_rc = self.rc(x2_upsample)  # 减少通道数
        shortcut = x2_rc

        x_cat = torch.cat((x1, x2_rc), dim=1)  # 拼接
        x_forward = self.rc2(x_cat)  # 减少通道数2
        x_forward = x_forward + shortcut
        pred = F.interpolate(self.predtrans(x_forward), TRAIN_SIZE, mode="bilinear", align_corners=True)  # 预测图

        return pred, x_forward


# Decoder





# if __name__ == '__main__':
#
#     input_rgb = torch.randn(2, 3, 256, 256)
#     input_d = torch.randn(2, 1, 256, 256)
#     net = MAG1()
#     out=net(input_rgb, input_d)
#     for out in out:
#      print("out1", out.shape)

#  Smeasure:0.862; maxEm:0.944; maxFm:0.907; MAE:0.0536.
if __name__ == '__main__':
    a = torch.randn(2, 3, 256, 256).cuda()
    b = torch.randn(2, 1, 256, 256).cuda()
    model = MAG1().cuda()
    #
    out = model(a, b)
    from FLOP import CalParams
    # flops, params = profile(model, inputs=(a,b))
    # flops, params = clever_format([flops, params], "%.2f")
    #  22.27M 15.20G
    CalParams(model,a,b)
    print('Total params %.2f'%(sum(p.numel() for p in model.parameters()) / 1000000.0))
#     print(out.shape) Elapsed time is 194.774440 seconds.
# [Statistics Information]
# FLOPs: 40.411G
# Params: 78.583M
#  ####################
# Total params 96.05
