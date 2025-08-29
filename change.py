
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from torch.utils.backcompat import keepdim_warning

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.ops import DeformConv2d


# class FeatureAlignmentModule(nn.Module):
#     def __init__(self, inc, outc):
#         super(FeatureAlignmentModule, self).__init__()
#
#         # 归一化层，提升稳定性
#         self.norm = nn.BatchNorm2d(inc)
#
#         # 使用 1x1 卷积调整通道数
#         self.conv1x1 = nn.Conv2d(inc, outc, kernel_size=1, stride=1, padding=0)
#
#         # 3x3 卷积用于特征变换，使其更易对齐
#         self.conv3x3 = nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=1)
#
#         # 可变形卷积（可选）
#
#         self.deform_conv = DeformConv2d(outc, outc, kernel_size=3, padding=1)
#         self.offset_conv = nn.Conv2d(outc, 18, kernel_size=3, padding=1)  # 18=2×3×3（2D offset）
#
#     def forward(self, x):
#         x = self.norm(x)  # 归一化
#         x = self.conv1x1(x)  # 通道变换
#
#         offset = self.offset_conv(x)
#         x = self.deform_conv(x, offset)  # 可变形卷积对齐
#
#         x = self.conv3x3(x)  # 进一步特征变换
#
#         return x
class SelfAttentionCNN(nn.Module):  #  #CNN
    def __init__(self, in_channels, num_heads=4):
        """
        Self-Attention Module for CNN
        :param in_channels: 输入通道数
        :param num_heads: 多头自注意力的头数
        """
        super(SelfAttentionCNN, self).__init__()
        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的缩放参数

    def forward(self, x):
        """
        :param x: 输入特征图 (batch_size, in_channels, H, W)
        :return: 经过自注意力增强的特征图
        """
        # Query, Key, Value
        x_q = rearrange(x, 'b c h w -> b c (h w)')
        x_k = rearrange(x, 'b c h w -> b (h w) c')
        attention1_map = F.sigmoid(torch.bmm(x_q, x_k))
        attention1_map = torch.mean(attention1_map, dim=2)
        attention_map = rearrange(attention1_map, 'b c -> b c 1 1')
        out = attention_map * x
        # 残差连接 + 可学习缩放
        out = self.gamma * out + x
        return out

class CNNPatchEmbedding(nn.Module):# vit
    def __init__(self, in_channels):#, embed_dim, patch_size=2
        """
        CNN-style Patch Embedding for ViT
        :param in_channels: 输入图像通道数 (e.g., 3 for RGB)
        :param embed_dim: ViT 需要的嵌入维度
        :param patch_size: Patch 大小 (e.g., 16 for ViT)
        """
        super(CNNPatchEmbedding, self).__init__()
        # self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.BatchNorm2d(in_channels)  # LayerNorm 替代 BN
        self.activation = nn.GELU()  # 可选激活函数
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 7, padding=3)

    def forward(self, x):
        """
        :param x: 输入图像 (batch_size, in_channels, H, W)
        :return: 处理后的 Patch Token (batch_size, num_patches, embed_dim)
        """
        # x = self.conv(x)  # (batch_size, embed_dim, H/P, W/P)
        # x = x.flatten(2).transpose(1, 2)  # 变换为 (batch_size, num_patches, embed_dim)
        # # print("x",x.shape)
        # x = self.norm(x)
        # x = self.activation(x).unsqueeze(1)
        x1 = self.conv1(x)
        x1_out =self.activation(self.norm(x1))
        x2 = self.conv2(x)
        x2_out = self.activation(self.norm(x2))
        x3 = self.conv3(x)
        x3_out = self.activation(self.norm(x3))
        out1 = self.fuse(x1_out,x2_out)
        out2 = self.fuse(out1,x3_out)
        return out2

    def fuse(self, f0, f1):
        max_0, _ = torch.max(f0, dim=1, keepdim=True)
        max_1, _ = torch.max(f1, dim=1, keepdim=True)
        add_max = max_0 + max_1
        add_max = F.sigmoid(add_max)
        return add_max*f0 +add_max*f1


def mi_loss(feat_s, feat_t, temperature=0.5):
    """
    Mutual Information Maximization Loss using InfoNCE.
    :param feat_s: Student model features (batch_size, feature_dim)
    :param feat_t: Teacher model features (batch_size, feature_dim)
    :param temperature: Temperature parameter for contrastive learning
    :return: MI loss
    """
    feat_s = feat_s.reshape(feat_s.size(0),-1)
    feat_t = feat_t.reshape(feat_s.size(0), -1)
    feat_s = F.normalize(feat_s, dim=-1)
    feat_t = F.normalize(feat_t, dim=-1)

    logits = torch.matmul(feat_s, feat_t.T) / temperature
    labels = torch.arange(logits.shape[0]).to(logits.device)

    loss = F.cross_entropy(logits, labels)
    return loss

def rkd_loss(feat_s, feat_t):#,a=1.0,b=1.0
    """
    Relational Knowledge Distillation Loss.
    :param feat_s: Student model features (batch_size, feature_dim)
    :param feat_t: Teacher model features (batch_size, feature_dim)
    :return: RKD loss
    """
    def pairwise_distance(x):
        """计算样本之间的欧几里得距离"""
        x_norm = F.normalize(x,p=2,dim=1)

        # dist = x_norm - 2.0 * torch.einsum('bij,bkj->bik',x_f, x_f) + x_norm.transpose(1, 2)

        return  torch.mm(x_norm,x_norm.t())

    feat_s = feat_s.reshape(feat_s.size(0), -1)
    feat_t = feat_t.reshape(feat_s.size(0), -1)
    p_s = pairwise_distance(feat_s)
    p_t = pairwise_distance(feat_t)
    r_l = F.mse_loss(feat_s,feat_t)
    d_l = F.mse_loss(p_s,p_t)
    # loss = r_l * a + d_l *b
    loss = r_l + d_l

    return loss


# class TM(nn.Module):
#     def __init__(self,inc,outc):
#         super().__init__()
#         self.conv = nn.Conv2d(inc,outc,kernel_size=1)
#         self.layer = nn.LayerNorm(outc)
#         self.inc = inc
#         self.outc = outc
#         self.v = CNNPatchEmbedding(outc)
#         self.c = SelfAttentionCNN(inc)
#         # self.conv1 =nn.Conv2d(512, 512, 1)
#         self.conv1 = nn.Conv2d(outc, outc * 2, kernel_size=1)
#         self.align = FeatureAlignmentModule(inc, outc)
#     def forward(self,ft,fm,alp=0.5,beta=0.5):
#         #修改tran的特征
#         b, c, h, w = fm.size()
#         f_v = self.conv(ft)
#         out = f_v.permute(0,2,3,1)
#         out = self.layer(out)
#         v_out = out.permute(0, 3, 1, 2)
#         # 计算相互损失
#
#         loss1 = rkd_loss(v_out, self.align(fm))
#         c_out= self.c(fm)
#         v_out = self.v(v_out)
#         loss2 = rkd_loss(c_out, v_out) # s
#         K = F.softmax(v_out) * c_out + v_out
#         return alp* loss1 + beta*loss2,F.interpolate(self.conv1(K), h//2, mode="bilinear", align_corners=True)
#
# class MT(nn.Module):
#     def __init__(self,inc,outc):
#         super().__init__()
#         self.linear = nn.Linear(inc,outc)
#         self.inc = inc
#         self.outc = outc
#         self.v = CNNPatchEmbedding(outc)
#         self.c = SelfAttentionCNN(inc)
#         self.align = FeatureAlignmentModule(inc, outc)
#         self.conv1 = nn.Conv2d(outc, outc * 2, kernel_size=1)
#     def forward(self,fm,ft,alp=0.3,beta=0.7):#
#         #首先是cnn转换到vit特征维度
#         b,c,h,w = fm.size()
#         fm = fm.permute(0, 2, 3, 1).reshape(-1, self.inc)
#         c_out = self.linear(fm).permute(1, 0).reshape(b, self.outc, h, w)
#         #计算相互损失
#         # loss1 = mi_loss(c_out, ft)
#         loss1 = rkd_loss(c_out, self.align(ft))
#         #进一步优势互补
#         c_out = self.c(c_out)
#         v_out = self.v(ft)
#         loss2 = rkd_loss(c_out,v_out)
#         K = F.softmax(v_out) * c_out + v_out
#         return alp* loss1 + beta*loss2,F.interpolate(self.conv1(K), h//2, mode="bilinear", align_corners=True)


class FeatureAlignmentModule(nn.Module):
    def __init__(self, inc, outc):
        super(FeatureAlignmentModule, self).__init__()

        # 归一化层，提升稳定性
        self.norm = nn.BatchNorm2d(inc)

        # 使用 1x1 卷积调整通道数
        self.conv1x1 = nn.Conv2d(inc, outc, kernel_size=1, stride=1, padding=0)

        # 3x3 卷积用于特征变换，使其更易对齐
        self.conv3x3 = nn.Conv2d(outc, outc, kernel_size=3, stride=1, padding=1)

        # 可变形卷积（可选）

        self.deform_conv = DeformConv2d(outc, outc, kernel_size=3, padding=1)
        self.offset_conv = nn.Conv2d(outc, 18, kernel_size=3, padding=1)  # 18=2×3×3（2D offset）

    def forward(self, x):
        x = self.norm(x)  # 归一化
        x = self.conv1x1(x)  # 通道变换
        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)  # 可变形卷积对齐
        x = self.conv3x3(x)  # 进一步特征变换

        return x


class At_loss(nn.Module):
    """Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer"""
    def __init__(self, p=2):
        super(At_loss, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        return self.at_loss(g_s, g_t)

    def at_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))

def gaussian_blur(x, kernel_size=5, sigma=1.0):
    """ 高斯模糊用于低频提取 """
    channels = x.shape[1]
    device = x.device
    x_coord = torch.arange(kernel_size, device=device).float() - (kernel_size - 1) / 2
    g_kernel = torch.exp(-0.5 * (x_coord[:, None]**2 + x_coord[None, :]**2) / sigma**2)
    g_kernel /= g_kernel.sum()
    g_kernel = g_kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return F.conv2d(x, g_kernel, padding=kernel_size // 2, groups=channels)


def laplacian_pyramid(x, levels=1):
    """ 使用金字塔方法进行特征分离 """
    # print("x", x.shape)
    current = x
    for _ in range(levels):
        blurred = gaussian_blur(current)
        downsampled = F.interpolate(blurred, scale_factor=0.5, mode='bilinear', align_corners=False)
        upsampled = F.interpolate(downsampled, size=x.shape[2:], mode='bilinear', align_corners=False)
        low_freq = upsampled
        high_freq = x - upsampled
    return low_freq, high_freq




class FeSep (nn.Module):
    def __init__(self,inc,outc):
        super().__init__()
        self.linear = nn.Linear(outc,outc)
        self.inc = outc
        self.gate = nn.Conv2d(outc, outc, kernel_size=1, bias=True)
        self.layer = nn.LayerNorm(outc)
        self.conv = nn.Conv2d(outc, outc, kernel_size=1)
        self.conv1 = nn.Conv2d(outc, outc*2, kernel_size=1)
        self.align = FeatureAlignmentModule(inc, outc)
        self.up2 = nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear')
        # self.ofd = OFD(inc, outc)
        self.f = CrossAttention(inc,outc)
        self.s =Similarity()
    def forward(self,fs,ft):#
        #映射头  s
        b,c,h,w = fs.size()
        fs = fs.permute(0, 2, 3, 1).reshape(-1, self.inc)
        c_out = self.linear(fs).permute(1, 0).reshape(b, self.inc, h, w)
        # 映射头  t
        # f_v = self.conv(ft)
        # out = f_v.permute(0, 2, 3, 1)
        # out = self.layer(out)
        # v_out = out.permute(0, 3, 1, 2)
        v_out = self.align(ft)
        loss2 = self.s(c_out, v_out)

        #特征增强
        #1
        fs_l,fs_h =laplacian_pyramid(c_out)
        #2
        ft_l, ft_h = laplacian_pyramid(v_out)
        alpha_A = torch.sigmoid(self.gate(fs_l))  # 计算每个像素的自适应权重 α
        alpha_B = torch.sigmoid(self.gate(ft_h))
        #自适应进行互补特征融合
        A_prime = alpha_A * fs_l + (1 - alpha_A) * ft_h
        B_prime = alpha_B * ft_h + (1 - alpha_B) * fs_l
        # A_prime = alpha_A * fs_l + (1 - alpha_A) * ft_h
        # B_prime = alpha_B * ft_l + (1 - alpha_B) * fs_h
        loss1 = rkd_loss(A_prime,B_prime)
        loss = loss1+loss2
        K = self.f(A_prime,B_prime) +self.f(B_prime,A_prime)
        return loss,F.interpolate(self.conv1(K), h//2, mode="bilinear", align_corners=True)#self.up2(self.conv1(K))#

    def laplacian_pyramid(x, levels=1):
        """ 使用金字塔方法进行特征分离 """
        # print("x",x)
        current = x
        for _ in range(levels):
            blurred = gaussian_blur(current)
            downsampled = F.interpolate(blurred, scale_factor=0.5, mode='bilinear', align_corners=False)
            upsampled = F.interpolate(downsampled, size=x.shape[2:], mode='bilinear', align_corners=False)
            low_freq = upsampled
            high_freq = x - upsampled
        return low_freq, high_freq

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



class FeSep1 (nn.Module):
    def __init__(self,inc,outc):
        super().__init__()
        self.linear = nn.Linear(outc,outc)
        self.inc = inc
        self.gate = nn.Conv2d(outc, outc, kernel_size=1, bias=True)
        self.layer = nn.LayerNorm(outc)
        self.conv = nn.Conv2d(outc, outc, kernel_size=1)
        self.conv1 = nn.Conv2d(outc, outc * 2, kernel_size=1)
        self.align = FeatureAlignmentModule(inc,outc)
        # self.ofd = OFD(inc,outc)
        self.up2 = nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear')
        self.f = CrossAttention(inc, outc)
        self.s =Similarity()
    def forward(self,ft,fs):

        #映射头  s
        b,c,h,w = fs.size()
        fs = fs.permute(0, 2, 3, 1).reshape(-1, self.inc)
        c_out = self.linear(fs).permute(1, 0).reshape(b, self.inc, h, w)

        # 映射头  t
        v_out = self.align(ft)
        loss2 = self.s(c_out,v_out)
        #低频，高频提取
        #1
        fs_l,fs_h =laplacian_pyramid(c_out)
        #2
        ft_l, ft_h = laplacian_pyramid(v_out)

        alpha_A = torch.sigmoid(self.gate(fs_l))  # 计算每个像素的自适应权重 α
        alpha_B = torch.sigmoid(self.gate(ft_l))
        #自适应进行互补特征融合
        A_prime = alpha_A * fs_l + (1 - alpha_A) * ft_h
        B_prime = alpha_B * fs_h + (1 - alpha_B) * ft_l
        loss1 = rkd_loss(A_prime,B_prime)
        loss = loss1+loss2
        # K = self.f(v_out,c_out)#F.softmax(v_out) * c_out +v_out
        K = self.f(A_prime, B_prime) + self.f(B_prime, A_prime)
        return loss,F.interpolate(self.conv1(K), h//2, mode="bilinear", align_corners=True) #self.up2(self.conv1(K))

    def laplacian_pyramid(x, levels=1):
        """ 使用金字塔方法进行特征分离 """

        current = x
        for _ in range(levels):
            blurred = gaussian_blur(current)
            downsampled = F.interpolate(blurred, scale_factor=0.5, mode='bilinear', align_corners=False)
            upsampled = F.interpolate(downsampled, size=x.shape[2:], mode='bilinear', align_corners=False)
            low_freq = upsampled
            high_freq = x - upsampled
        return low_freq, high_freq

# if __name__ == "__main__":
#     input_rgb = torch.randn(2, 512, 8, 8)
#     input_d = torch.randn(2, 512, 8, 8)
#     t = torch.randn(2, 768, 8, 8)
#
#     # net = Adfusion(768,768)
#     # net = MultiLevelDynamicCrossAttentionFusion(768,768)
#     # net = FeatureAlignmentModule(768,512)
#     # net = FeSep1(768,512)
#     net = FeSep(512, 512)
#     out = net(input_rgb,input_d)
#     # out = net(t)
#     # out = net(t)
#     for out in out:
#         print("out1", out.shape)


class ATT(nn.Module):
    def __init__(self, channel):
        super(ATD1, self).__init__()
        self.k = 64
        # 定义线性变换层
        self.linear_s = nn.Sequential(
            nn.Conv1d(channel, self.k, 1, bias=False),
            nn.Conv1d(channel, self.k, 1, bias=False)
        )
        self.linear_re_s = nn.Conv1d(self.k, channel, 1, bias=False)
        self.linear_t = nn.Sequential(
            nn.Conv1d(channel, self.k, 1, bias=False),
            nn.Conv1d(channel, self.k, 1, bias=False)
        )
        self.linear_re_t = nn.Conv1d(self.k, channel, 1, bias=False)
        self.attention_loss = At_loss()
    def _attention_projection(self, x, linear, linear_re):
        b, c, h, w = x.shape
        x = x.view(b, c, -1)  # 变换形状以匹配 Conv1d

        softmax_x = F.softmax(linear[0](x), dim=-1)
        linear_project_x = linear[1](x)

        at_x = linear_re(linear_project_x * softmax_x)
        return at_x.view(b, c, h, w) + x.view(b, c, h, w)

    def forward(self, student, teacher):
        at_s = self._attention_projection(student, self.linear_s, self.linear_re_s)
        at_t = self._attention_projection(teacher, self.linear_t, self.linear_re_t)
        # k = at_s  + at_t
        loss_attention = self.attention_loss(at_s, at_t)
        return loss_attention


# if __name__ == '__main__':
#
#     input_rgb = torch.randn(2, 1, 256, 256)
#     input_d = torch.randn(2, 1, 256, 256)
#     net = ATD1(1)
#     out=net(input_rgb, input_d)
#     # for out in out:
#     print("out1", out)

class KLDLoss(nn.Module):
    def __init__(self, alpha=1, tau=1, resize_config=None, shuffle_config=None, transform_config=None,
                 warmup_config=None, earlydecay_config=None):
        super().__init__()
        self.alpha_0 = alpha
        self.alpha = alpha
        self.tau = tau

        self.resize_config = resize_config
        self.shuffle_config = shuffle_config
        self.transform_config = transform_config
        self.warmup_config = warmup_config
        self.earlydecay_config = earlydecay_config

        self.KLD = torch.nn.KLDivLoss(reduction='sum')

    def resize(self, x, gt):#调整特征尺寸
        mode = self.resize_config['mode']
        align_corners = self.resize_config['align_corners']
        x = F.interpolate(
            input=x,
            size=gt.shape[2:],
            mode=mode,
            align_corners=align_corners)
        return x

    def shuffle(self, x_student, x_teacher, n_iter):#通道洗牌
        interval = self.shuffle_config['interval']
        print(interval, "1")
        B, C, W, H = x_student.shape
        if n_iter % interval == 0:
            print("2")
            idx = torch.randperm(C)
            x_student = x_student[:, idx, :, :].contiguous()
            x_teacher = x_teacher[:, idx, :, :].contiguous()
        print("3")
        return x_student, x_teacher

    def transform(self, x):#特征变换
        B, C, W, H = x.shape
        loss_type = self.transform_config['loss_type']
        if loss_type == 'pixel':
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(B, W * H, C)
        elif loss_type == 'channel':
            group_size = self.transform_config['group_size']
            if C % group_size == 0:
                x = x.reshape(B, C // group_size, -1)
            else:
                n = group_size - C % group_size
                x_pad = -1e9 * torch.ones(B, n, W, H).cuda()
                x = torch.cat([x, x_pad], dim=1)
                x = x.reshape(B, (C + n) // group_size, -1)
        return x

    def warmup(self, n_iter):#学习率预热
        # print("war")
        mode = self.warmup_config['mode']
        warmup_iters = self.warmup_config['warmup_iters']
        if n_iter > warmup_iters:
            return
        elif n_iter == warmup_iters:
            self.alpha = self.alpha_0
            return
        else:
            if mode == 'linear':
                self.alpha = self.alpha_0 * (n_iter / warmup_iters)
            elif mode == 'exp':
                self.alpha = self.alpha_0 ** (n_iter / warmup_iters)
            elif mode == 'jump':
                self.alpha = 0

    def earlydecay(self, n_iter):#早衰
        mode = self.earlydecay_config['mode']
        earlydecay_start = self.earlydecay_config['earlydecay_start']
        earlydecay_end = self.earlydecay_config['earlydecay_end']

        if n_iter < earlydecay_start:
            return
        elif n_iter > earlydecay_start and n_iter < earlydecay_end:
            if mode == 'linear':
                self.alpha = self.alpha_0 * ((earlydecay_end - n_iter) / (earlydecay_end - earlydecay_start))
            elif mode == 'exp':
                self.alpha = 0.001 * self.alpha_0 ** ((earlydecay_end - n_iter) / (earlydecay_end - earlydecay_start))
            elif mode == 'jump':
                self.alpha = 0
        elif n_iter >= earlydecay_end:
            self.alpha = 0

    def forward(self, x_student, x_teacher):#, gt, n_iter
        # print("start kld")
        if self.warmup_config:
            print("warm")
            self.warmup(n_iter)
        if self.earlydecay_config:
            print("decay")
            self.earlydecay(n_iter)

        if self.resize_config:
            print("resize(")
            x_student, x_teacher = self.resize(x_student, gt), self.resize(x_teacher, gt)
        if self.shuffle_config:
            print("shuffle")
            x_student, x_teacher = self.shuffle(x_student, x_teacher, n_iter)
        if self.transform_config:
            print("transform")
            x_student, x_teacher = self.transform(x_student), self.transform(x_teacher)


        x_student = F.log_softmax(x_student / self.tau, dim=-1)
        x_teacher = F.softmax(x_teacher / self.tau, dim=-1)
        loss = self.KLD(x_student, x_teacher) / (x_student.numel() / x_student.shape[-1])
        # print("self.alpha", self.alpha)
        loss = self.alpha * loss
        return loss

#中间特征蒸馏
class OFD(nn.Module):#用于特征蒸馏
    '''
	A Comprehensive Overhaul of Feature Distillation
	http://openaccess.thecvf.com/content_ICCV_2019/papers/
	Heo_A_Comprehensive_Overhaul_of_Feature_Distillation_ICCV_2019_paper.pdf
	'''

    def __init__(self, in_channels, out_channels):
        super(OFD, self).__init__()
        self.connector = nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, fm_s, fm_t):
        margin = self.get_margin(fm_t)
        fm_t = torch.max(fm_t, margin)
        fm_s = self.connector(fm_s)

        mask = 1.0 - ((fm_s <= fm_t) & (fm_t <= 0.0)).float()
        loss = torch.mean((fm_s - fm_t) ** 2 * mask)

        return loss

class Similarity(nn.Module):
    ##Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author##
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, g_s, g_t):
        return self.similarity_loss(g_s, g_t)

    def similarity_loss(self, f_s, f_t):
        # print('f_s', f_s.shape)
        bsz = f_s.shape[0]
        # print('bsz', bsz)
        f_s = f_s.reshape(bsz, -1)
        f_t = f_t.reshape(bsz, -1)
        # print('f_s', f_s.shape)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)
        # print('G_t', G_t.shape)
        G_diff = G_t - G_s
        # loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        loss = (G_diff * G_diff).view(-1, 1).sum(0)
        # print('(G_diff * G_diff).view(-1, 1)', (G_diff * G_diff).view(-1, 1).shape)
        return loss


def hcl(fstudent, fteacher):
    loss_all = 0.0
    B, C, h, w = fstudent.size()
    loss = F.mse_loss(fstudent, fteacher, reduction='mean')
    cnt = 1.0
    tot = 1.0
    for l in [4,2,1]:
        if l >=h:
            continue
        tmpfs = F.adaptive_avg_pool2d(fstudent, (l,l))
        tmpft = F.adaptive_avg_pool2d(fteacher, (l,l))
        cnt /= 2.0
        loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
        tot += cnt
    loss = loss / tot
    loss_all = loss_all + loss
    return loss_all

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


class CrossAttention(nn.Module):
    def __init__(self, input_size=256, output_size=256, linear_dim=256):
        super(CrossAttention, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels=input_size, out_channels=output_size, kernel_size=1)
        self.conv1x1_1 = nn.Conv2d(in_channels=output_size, out_channels=input_size, kernel_size=1)
        self.linear_dim = linear_dim
        self.output_size = output_size

        # 修改 linear 层的维度以匹配 16*16=256
        self.linear_query = torch.nn.Linear(input_size, output_size)  # 16*16 = 256
        self.linear_key = torch.nn.Linear(output_size, output_size)
        self.linear_value = torch.nn.Linear(output_size, output_size)

        self.conv_forget_attended = nn.Conv1d(output_size, output_size, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.forget_gate_linear = torch.nn.Linear(output_size, output_size)

    def forward(self, feature1, feature2):
        # feature1, feature2: [2, 256, 16, 16]
        B = feature1.shape[0]  # 2
        H = feature1.shape[2]  # 16
        W = feature1.shape[3]  # 16

        # 1x1 卷积保持空间维度不变
        feature1_conv1 = self.conv1x1(feature1)  # [2, 256, 16, 16]
        feature2_conv1 = self.conv1x1(feature2)  # [2, 256, 16, 16]

        # 重排维度以进行注意力计算
        # 将特征图展平为序列形式
        q_tensor1 = feature1_conv1.view(B, self.output_size, -1)  # [2, 256, 256]
        k_tensor2 = feature2_conv1.view(B, self.output_size, -1)  # [2, 256, 256]
        v_tensor2 = feature2_conv1.view(B, self.output_size, -1)  # [2, 256, 256]

        # 应用线性变换
        # print(q_tensor1.shape, q_tensor1.transpose(1, 2).shape)
        q_tensor1 = self.linear_query(q_tensor1.transpose(1, 2)).transpose(1, 2)  # [2, 256, 256]
        k_tensor2 = self.linear_key(k_tensor2.transpose(1, 2)).transpose(1, 2)  # [2, 256, 256]
        v_tensor2 = self.linear_value(v_tensor2.transpose(1, 2)).transpose(1, 2)  # [2, 256, 256]

        # 计算注意力
        attention_scores = torch.matmul(q_tensor1, k_tensor2.transpose(-2, -1)) / (self.linear_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        output_tensor = torch.matmul(attention_weights, v_tensor2)  # [2, 256, 256]

        # 应用遗忘门
        forget_gate = self.forget_gate_linear(output_tensor.transpose(1, 2)).transpose(1, 2)
        forget_gate = torch.sigmoid(forget_gate)

        # 重塑回原始维度
        output_tensor = output_tensor.view(B, self.output_size, H, W)
        forget_gate = forget_gate.view(B, self.output_size, H, W)

        # 最终输出
        output_tensor = self.conv1x1_1(output_tensor * forget_gate + feature2_conv1)  # [2, 256, 16, 16]

        return output_tensor
class CrossAttention(nn.Module):
    def __init__(self, in_channels, heads=4):
        super(CrossAttention, self).__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.dim_head = in_channels // heads

        # Q, K, V projections
        self.to_q = nn.Linear(in_channels, in_channels)
        self.to_k = nn.Linear(in_channels, in_channels)
        self.to_v = nn.Linear(in_channels, in_channels)

        # 输出融合线性层
        self.to_out = nn.Linear(in_channels, in_channels)

    def forward(self, feat_A, feat_B):
        B, C, H, W = feat_A.shape
        N = H * W

        # 展平为序列 [B, C, H*W] -> [B, H*W, C]
        feat_A_flat = feat_A.view(B, C, -1).permute(0, 2, 1) # Query
        feat_B_flat = feat_B.view(B, C, -1).permute(0, 2, 1) # Key, Value

        # Q, K, V
        Q = self.to_q(feat_A_flat) # [B, N, C]
        K = self.to_k(feat_B_flat)
        V = self.to_v(feat_B_flat)

        # 交叉注意力计算
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim_head ** 0.5) # [B, N, N]
        attn_weights = F.softmax(attn_scores, dim=-1) # [B, N, N]

        out = torch.matmul(attn_weights, V) # [B, N, C]
        out = self.to_out(out) # [B, N, C]

# 恢复空间形状
        out = out.permute(0, 2, 1).view(B, C, H, W)

        return out
# import numpy as np from typing import Tuple, Optional
class AdaptiveRectificationModule(nn.Module):
    """ Adaptive Rectification Module (ARM) for RLCL 用于自适应校正预测结果 """
    def __init__(self, temperature: float = 1.0):#, num_classes: int
        super().__init__()
        # self.num_classes = num_classes
        self.temperature = temperature
    def forward(self, logits: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """ Args: logits: 模型预测的logits [B, C, H, W] ground_truth: 真实标签 [B, H, W] Returns: rectified_logits: 校正后的logits """
        # 计算
        soft_labels = F.softmax(logits / self.temperature, dim=1) #
        #创建ground truth的one-hot编码
        gt_one_hot = ground_truth.float()
        # 自适应校正：使用lambda动态调整
        lambda_weight = self._compute_adaptive_weight(logits, gt_one_hot)
        # 校正公式：rectified = lambda * gt + (1-lambda) * soft_pred

        rectified_logits = lambda_weight * gt_one_hot + (1 - lambda_weight) * soft_labels

        return rectified_logits
    def _compute_adaptive_weight(self, logits: torch.Tensor, gt_one_hot: torch.Tensor) -> torch.Tensor:
        """计算自适应权重lambda"""
        # 计算预测概率
        pred_probs = F.softmax(logits, dim=1) #
        # 计算置信度 (预测正确类别的概率)
        confidence = torch.sum(pred_probs * gt_one_hot, dim=1, keepdim=True)
        # 自适应lambda: 置信度越低，越依赖ground truth
        lambda_weight = 1.0 - confidence

        return lambda_weight
# class RLCLLoss(nn.Module):
#     """ RLCL损失函数：Logit-level的互学习损失 """
#     def __init__(self, alpha: float = 0.5, temperature: float = 3.0):
#         super().__init__()
#         self.alpha = alpha
#         self.temperature = temperature
#         self.kl_loss = nn.KLDivLoss(reduction='batchmean')
#     def forward(self, logits_main: torch.Tensor, logits_aux: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         """ Args: logits_main: 主分支的logits logits_aux: 辅助分支的logits targets: 真实标签 """
#         # 标准交叉熵损失
#         ce_loss_main = F.cross_entropy(logits_main, targets)
#         ce_loss_aux = F.cross_entropy(logits_aux, targets)
#         # Logit-wise KL散度损失（双向）
#         soft_main = F.log_softmax(logits_main / self.temperature, dim=1)
#         soft_aux = F.log_softmax(logits_aux / self.temperature, dim=1)
#         target_main = F.softmax(logits_aux / self.temperature, dim=1)
#         target_aux = F.softmax(logits_main / self.temperature, dim=1)
#         kl_loss_main = self.kl_loss(soft_main, target_aux)
#         kl_loss_aux = self.kl_loss(soft_aux, target_main) #
#         # 总损失
#         total_loss = (ce_loss_main + ce_loss_aux) + self.alpha * (kl_loss_main + kl_loss_aux) * (self.temperature ** 2)
#
#         return total_loss












if __name__ == '__main__':

    input_rgb = torch.randn(2, 256, 16, 16)
    input_d = torch.randn(2, 256, 16, 16)
    net = CrossAttention(256)
    out=net(input_rgb, input_d)
    # for out in out:
    print("out1", out.shape)

    # linear_query = torch.nn.Linear(64, 256)
    # x = torch.randn(2, 512, 64)
    # print(linear_query(x).shape)
# Net1
# /media/pc12/data/LYQ/model/train_test1/Pth2/Net1_12_2025_04_10_15_33_last.pth20
# Smeasure:0.853; maxEm:0.936; maxFm:0.897; MAE:0.0590.


#
#  epoch: 66  || best mae:0.05793657567537292
# 2025-04-11 20:33:31 | Epoch:179/200 || trainloss:0.29083899 valloss:0.27354744 || valmae:0.05427265 || lr_rate:3.90625e-07 || spend_time:0:3:56
# =======best mae epoch:123,best mae:0.054033919160372645
# 2025-04-11 20:33:33 | best mae epoch:123  || best mae:0.054033919160372645



# MDmodel1_443
# 2025-04-14 08:10:01 | best mae epoch: 60  || best mae:0.05689761892619712
# 2025-04-14 08:10:01 | Epoch:200/200 || trainloss:0.33380809 valloss:0.27510179 || valmae:0.05514953 || lr_rate:9.765625e-08 || spend_time:0:3:59
# =======best mae epoch:115,best mae:0.054537955249623696
# 2025-04-14 08:10:04 | best mae epoch:115  || best mae:0.054537955249623696