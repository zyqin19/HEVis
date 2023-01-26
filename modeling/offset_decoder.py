import torch
import torch.nn as nn
import math
import numpy as np
from stemseg.modeling.common import UpsampleTrilinear3D
from stemseg.config import cfg


class OffsetDecoder(nn.Module):
    def __init__(self, conv1_in_channels, conv1_output_channels, down_sample=2, up='upsample', plus='addition'):
        super().__init__()

        # Reduce the number of channels to the embedding dimension
        self.conv1_add = nn.Sequential(
            nn.Conv3d(conv1_in_channels, conv1_in_channels//4, 1, stride=1),
            nn.Conv3d(conv1_in_channels//4, conv1_output_channels, 1, stride=1),
            # nn.ReLU(inplace=True),
        )

        self.conv1_cat = nn.Sequential(
            nn.Conv3d(conv1_in_channels * 4, conv1_in_channels, 1, stride=1),
            nn.Conv3d(conv1_in_channels, conv1_in_channels // 4, 1, stride=1),
            nn.Conv3d(conv1_in_channels // 4, conv1_output_channels, 1, stride=1),
            # nn.ReLU(inplace=True),
        )

        # Reduce the number of channels to 3
        # self.conv2 = nn.Sequential(
        #     nn.Conv3d(conv1_output_channels, 3, 1, stride=1),  # （3，X， X）
        # )

        self.pooling_blocks = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2)),
            nn.AvgPool3d(kernel_size=(1, 2, 2)),
        )

        # if down_sample > 1:
        #     for i in range(int(math.log2(down_sample))):
        #         self.pooling_blocks.append(nn.Conv3d(conv1_output_channels, conv1_output_channels, 1, stride=1))
        #         self.pooling_blocks.append(nn.AvgPool3d(kernel_size=(1, 2, 2)))

        self.up = up
        self.plus = plus
        self.plus_weight = cfg.MODEL.OFFSET.FEATURE_WEIGHT
        self.n_bits = cfg.MODEL.OFFSET.N_BITS

        # offset 32x -> 16x
        self.upsample_32_to_16 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(1, 2, 2), align_corners=False)
        )
        # offset 16x to 8x
        self.upsample_16_to_8 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(1, 2, 2), align_corners=False)
        )
        # offset 8x to 4x
        self.upsample_8_to_4 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(1, 2, 2), align_corners=False)
        )
        # offset 32x to 4x
        self.upsample_32_to_4 = nn.Sequential(
            UpsampleTrilinear3D(scale_factor=(1, 8, 8), align_corners=False)
        )

    def forward(self, input):
        assert len(input) == 4, "Expected 4 feature maps, got {}".format(len(input))
        n_bins = 2.0 ** self.n_bits
        feat_map_32x, feat_map_16x, feat_map_8x, feat_map_4x = input

        x = self.upsample_32_to_4(feat_map_32x)
        offset = self.conv1_add(x)

        out = self.pooling_blocks(offset)

        out = torch.sigmoid(out)
        out = out * 255
        if self.n_bits < 8:
            out = torch.floor(out / 2 ** (8 - self.n_bits))
        out = out / n_bins - 0.5
        # torch.cuda.empty_cache()

        return offset, out

