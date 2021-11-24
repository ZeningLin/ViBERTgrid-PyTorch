import torch
import torch.nn as nn

from typing import Any, Tuple


class ROIEmbedding(nn.Module):
    def __init__(self, num_channels: int, roi_shape: Any) -> None:
        super().__init__()

        if isinstance(roi_shape, Tuple):
            assert len(
                roi_shape) == 2, f"roi_shape must be int or two-element tuple, {len(roi_shape)} elements were given"
            num_flatten = num_channels * roi_shape[0] * roi_shape[1]
        elif isinstance(roi_shape, int):
            num_flatten = num_channels * roi_shape * roi_shape
        else:
            raise ValueError("roi_shape must be int or two-element tuple")

        self.conv_1 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(num_channels)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(num_channels)
        self.relu_2 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(num_flatten, 1024)

    def forward(self, ROI: torch.Tensor) -> torch.Tensor:
        # ROI feature map -> 1024-d feature vector
        ROI_emb = self.conv_1(ROI)
        ROI_emb = self.bn_1(ROI_emb)
        ROI_emb = self.relu_1(ROI_emb)
        ROI_emb = self.conv_2(ROI_emb)
        ROI_emb = self.bn_2(ROI_emb)
        ROI_emb = self.relu_2(ROI_emb)
        ROI_emb = self.flatten(ROI_emb)
        ROI_emb = self.linear(ROI_emb)

        return ROI_emb
