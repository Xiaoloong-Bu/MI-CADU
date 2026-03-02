import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math

from torch.nn import LeakyReLU


class GuidBlock(nn.Module):
    def __init__(self, dim4, dim1, out_channel=256):
        super(GuidBlock, self).__init__()

        # 高维特征分类
        self.conv_1 = nn.Sequential(
            DWConvBlock(in_channels=dim4, out_channels=dim4 * 2, kernel_size=3, stride=1, padding=1),
            Conv1x1Block(in_channels=dim4 * 2, out_channels=dim4 * 2),
            DWConvBlock(in_channels=dim4 * 2, out_channels=dim4, kernel_size=3, stride=1, padding=1),
            Conv1x1Block(in_channels=dim4, out_channels=dim4),
        )
        self.conv_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim1, out_channels=dim1, kernel_size=3, stride=1, bias=False),
            nn.ReLU(),
        )
        self.conv_3 = nn.Conv2d(in_channels=dim1, out_channels=1, kernel_size=1, stride=1, bias=False)
        self.cls_1 = ClassificationAttention(out_channel)

        # 低维特征对比学习
        self.linear_d_1 = nn.Sequential(
            nn.Linear(256, 256 * 2),
            nn.LeakyReLU(),
            nn.Linear(256 * 2, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )

        self.linear_1 = nn.Sequential(
            nn.Linear(out_channel * 2, out_channel),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_channel, out_channel),
            nn.LeakyReLU(),
        )
        self.fc = nn.Linear(out_channel, 11)

        self.fc_linear = nn.Sequential(
            nn.Linear(11, out_channel * 2),
            nn.LeakyReLU(),
            nn.Linear(out_channel * 2, out_channel),
            nn.LeakyReLU(),
        )

    def split_block(self, x):
        B, C, H, W = x.size()
        padw = padh = 0
        if H % 16 != 0:
            padh = 16 - H % 16
        if W % 16 != 0:
            padw = 16 - W % 16
        x = nn.ReplicationPad2d((padw, 0, padh, 0))(x)
        patch_x = x.view(B, 1, H // 16, 16, W // 16, 16)
        patch_x = patch_x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, 256)
        return patch_x, H, W, padw, padh

    def forward(self, img_1, img_4, hu_guid=None, t_cls=None):
        # 高维特征分类
        img_4 = self.conv_1(img_4)
        cls_feature = self.cls_1(img_4)
        # 低维特征对比学习
        t_img_1 = self.conv_2(img_1)
        img_1 = self.conv_3(t_img_1)  # 变成一个通道
        # 分块
        patch_1, H, W, padw, padh = self.split_block(img_1)
        patch_1 = self.linear_d_1(patch_1) + patch_1

        features = torch.cat([cls_feature, patch_1.mean(1)], dim=1)
        x = self.linear_1(features)
        out = self.fc(x)

        if hu_guid is not None:
            guid_features = None
            for i in range(len(hu_guid)):
                if guid_features is None:
                    guid_features = self.fc_linear(F.normalize(hu_guid[i], dim=-1))
                else:
                    guid_features += self.fc_linear(F.normalize(hu_guid[i], dim=-1))
        else:
            if t_cls is not None:
                guid_features = self.fc_linear(F.normalize(t_cls, dim=-1))
            else:
                guid_features = self.fc_linear(F.normalize(out, dim=-1))

        return out, guid_features


class Cross_attention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)

        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)

        self.dwconv_1 = DWConvBlock(in_channel, in_channel, 3, 1, 1)
        self.dwconv_2 = DWConvBlock(in_channel, in_channel, 5, 1, 2)

    def forward(self, inputs):
        x = self.dwconv_1(inputs)
        y = self.dwconv_2(inputs)
        batch, channel, height, width = x.shape

        n_head = self.n_head
        head_dim = channel // n_head

        x_A = self.norm_A(x)
        qkv_A = self.qkv_A(x_A).view(batch, n_head, head_dim * 3, height, width)
        query_A, key_A, value_A = qkv_A.chunk(3, dim=2)

        x_B = self.norm_B(y)
        qkv_B = self.qkv_B(x_B).view(batch, n_head, head_dim * 3, height, width)
        query_B, key_B, value_B = qkv_B.chunk(3, dim=2)

        attn_A = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_B, key_A
        ).contiguous() / math.sqrt(channel)
        attn_A = attn_A.view(batch, n_head, height, width, -1)
        attn_A = torch.softmax(attn_A, -1)
        attn_A = attn_A.view(batch, n_head, height, width, height, width)

        out_A = torch.einsum("bnhwyx, bncyx -> bnchw", attn_A, value_A).contiguous()
        out_A = self.out_A(out_A.view(batch, channel, height, width))
        out_A = out_A + x_A

        attn_B = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_A, key_B
        ).contiguous() / math.sqrt(channel)
        attn_B = attn_B.view(batch, n_head, height, width, -1)
        attn_B = torch.softmax(attn_B, -1)
        attn_B = attn_B.view(batch, n_head, height, width, height, width)

        out_B = torch.einsum("bnhwyx, bncyx -> bnchw", attn_B, value_B).contiguous()
        out_B = self.out_B(out_B.view(batch, channel, height, width))
        out_B = out_B + x_B
        return torch.cat((out_A, out_B), 1)


class ClassificationAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()

        self.n_head = n_head
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Linear(in_channel, in_channel * 3)
        self.out = nn.Linear(in_channel, in_channel)

        # 添加分类token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, in_channel))

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        # 准备分类token并扩展到batch大小
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        # 归一化输入
        norm = self.norm(input)
        # 展平空间维度并添加分类token
        norm_flat = norm.view(batch, channel, -1).transpose(1, 2)
        norm_with_cls = torch.cat([cls_tokens, norm_flat], dim=1)
        # 计算QKV
        qkv_with_cls = self.qkv(norm_with_cls).transpose(1, 2)
        qkv_with_cls = qkv_with_cls.view(batch, n_head, head_dim * 3, -1)
        query, key, value = qkv_with_cls.chunk(3, dim=2)
        # 计算注意力分数
        query_transposed = query.transpose(2, 3)
        attn_flat = torch.matmul(query_transposed, key) / math.sqrt(head_dim)
        # 应用softmax
        attn_flat = torch.softmax(attn_flat, dim=-1)
        # 计算输出
        out_flat = torch.matmul(attn_flat, value.transpose(2, 3))
        out_flat = out_flat.transpose(2, 3).view(batch, channel, -1)

        cls_out = out_flat[:, :, 0:1].view(batch, head_dim)  # [batch, n_head, head_dim, 1]
        return self.out(cls_out)


class Attention_spatial(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)
        self.relu = nn.ReLU()

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(head_dim)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


## Feature Modulation
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True, up_num=1):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level)),
        )

    def forward(self, x, embed, use_cls=True):
        batch, C, H, W = x.shape
        if use_cls:
            # 通道
            gamma, beta = self.MLP(embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        return x


class OuterProductSpatialGuidance(nn.Module):
    def __init__(self, guid_dim, feat_channels):
        super().__init__()

        self.channel_affine = nn.Sequential(
            nn.Linear(guid_dim, guid_dim * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(guid_dim * 2, feat_channels * 2)
        )

        self.conv = nn.Sequential(
            DWConvBlock(feat_channels, feat_channels, kernel_size=3, padding=1, stride=1),
            Conv1x1Block(feat_channels, feat_channels),
        )

    def forward(self, feat, guid, use_cls=True):
        feat = self.conv(feat)
        if not use_cls:
            return feat

        B, C, H, W = feat.shape
        gamma, beta = self.channel_affine(guid).chunk(2, dim=1)
        gamma = gamma.view(B, C, 1, 1)
        beta  = beta.view(B, C, 1, 1)
        feat_c = gamma * feat + beta

        return feat_c + feat


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv_1(x)
        return x


class DWConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      groups=min(in_channels, out_channels), bias=bias),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv_1(x)
        return x


class Conv1x1Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv_1(x)
        return x


class OutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv_1(x)
        return x


# SE注意力模块
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.channels = channels
        self.reduction = reduction

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1, 1)
        return x * y


class Encoder(nn.Module):
    def __init__(self, inp_channels=2, dim=32):
        super().__init__()

        self.encoder_level1 = nn.Sequential(
            ConvBlock(in_channels=inp_channels, out_channels=dim, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
        )

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 1
        self.encoder_level2 = nn.Sequential(
            nn.GroupNorm(16, dim * 2 ** 1),
            Block(dim * 2 ** 1),
        )

        self.down2_3 = Downsample(dim * 2 ** 1)  ## From Level 1 to Level 3
        self.encoder_level3 = nn.Sequential(
            nn.GroupNorm(16, dim * 2 ** 2),
            Block(dim * 2 ** 2),
        )

        self.down3_4 = Downsample(dim * 2 ** 2)  ## From Level 3 to Level 4
        self.encoder_level4 = nn.Sequential(
            nn.GroupNorm(16, dim * 2 ** 3),
            Block(dim * 2 ** 3),
        )

    def forward(self, inp_img):
        out_enc_level1 = self.encoder_level1(inp_img)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        out_enc_level4 = self.encoder_level4(inp_enc_level4)

        return out_enc_level4, out_enc_level3, out_enc_level2, out_enc_level1


class Process_Embed(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super().__init__()

        self.fusion_proj = Conv1x1Block(embed_dim * 2, embed_dim)

    def forward(self, x):
        x = self.fusion_proj(x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Block(nn.Module):
    def __init__(self, channel):
        super(Block, self).__init__()
        self.conv_1 = nn.Sequential(
            ConvBlock(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1),
            SEBlock(channel, reduction=8),
            ConvBlock(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1)
        )
    def forward(self, x):
        return self.conv_1(x)


class CAUD(nn.Module):
    img_lst = []

    def __init__(self, inp_channels=2, dim=32):
        super().__init__()

        self.encoder = Encoder(inp_channels, dim)

        self.guid = GuidBlock(dim * 2 ** 3, dim, 256)

        self.cross_attention1 = Cross_attention(dim * 2 ** 3)
        self.attention_spatial = Attention_spatial(dim * 2 ** 3)

        self.feature_process = Process_Embed(dim * 2 ** 3)
        self.guidance_4 = OuterProductSpatialGuidance(256, dim * 2 ** 3)
        self.decoder_level4 = nn.Sequential(
            nn.GroupNorm(16, dim * 2 ** 3),
            Block(dim * 2 ** 3),
        )

        self.guidance_3 = OuterProductSpatialGuidance(256, dim * 2 ** 2)
        self.up4_3 = Upsample(dim * 2 ** 3)  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 3, dim * 2 ** 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.decoder_level3 = nn.Sequential(
            nn.GroupNorm(16, dim * 2 ** 2),
            Block(dim * 2 ** 2),
        )

        self.guidance_2 = OuterProductSpatialGuidance(256, dim * 2 ** 1)
        self.up3_2 = Upsample(dim * 2 ** 2)  ## From Level 3 to Level 1
        self.reduce_chan_level2 = nn.Sequential(
            nn.Conv2d(dim * 2 ** 2, dim * 2 ** 1, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.decoder_level2 = nn.Sequential(
            nn.GroupNorm(16, dim * 2 ** 1),
            Block(dim * 2 ** 1),
        )

        self.guidance_1 = OuterProductSpatialGuidance(256, dim)
        self.up2_1 = Upsample(dim * 2 ** 1)  ## From Level 1 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(
            nn.GroupNorm(16, dim * 2 ** 1),
            Block(dim * 2 ** 1),
        )

        self.refinement = nn.Sequential(
            Block(dim * 2 ** 1),
        )

        self.output = nn.Sequential(
            nn.Conv2d(dim * 2 ** 1, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, inp_img_A, inp_img_B, hu_guid=None, use_cls=True, t_cls=None):
        out_enc_level4, out_enc_level3, out_enc_level2, out_enc_level1 = self.encoder(torch.cat((inp_img_A, inp_img_B), dim=1))

        cls_out, guid_features = self.guid(out_enc_level1, out_enc_level4, hu_guid, t_cls)

        # 融合阶段
        out_enc_level = self.cross_attention1(out_enc_level4)
        out_enc_level = self.feature_process(out_enc_level)
        out_enc_level = self.attention_spatial(out_enc_level)

        # 解码阶段
        out_enc_level = self.guidance_4(out_enc_level, guid_features, use_cls)

        out_enc_level = self.decoder_level4(out_enc_level)

        inp_dec_level3 = self.up4_3(out_enc_level)
        inp_dec_level3 = self.guidance_3(inp_dec_level3, guid_features, use_cls)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = self.guidance_2(inp_dec_level2, guid_features, use_cls)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = self.guidance_1(inp_dec_level1, guid_features, use_cls)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1, cls_out
