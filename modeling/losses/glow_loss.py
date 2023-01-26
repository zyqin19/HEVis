import torch
import torch.nn as nn
from stemseg.utils import ModelOutputConsts, LossConsts
from math import log
from stemseg.config import cfg

class GlowLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_lovasz = cfg.TRAINING.LOSSES.WEIGHT_OFFSET

    # def forward(self, nll, output_dict):
    #     output_dict[ModelOutput.OTHERS][LossConsts.OFFSET_LOSS] = torch.abs(torch.mean(nll))
    def forward(self, log_p, logdet, n_pixel, n_bins, output_dict):
        # n_pixel = image_size * image_size * in_channel

        loss = -log(n_bins) * n_pixel
        loss = loss + logdet + log_p

        loss = torch.abs((-loss / (log(2) * n_pixel)).mean())

        output_dict[ModelOutputConsts.OTHERS][LossConsts.OFFSET_LOSS] = loss
        # output_dict[ModelOutputConsts.OPTIMIZATION_LOSSES][LossConsts.OFFSET_LOSS] = loss * cfg.TRAINING.LOSSES.\
        #                                                                                 WEIGHT_OFFSET

        return loss