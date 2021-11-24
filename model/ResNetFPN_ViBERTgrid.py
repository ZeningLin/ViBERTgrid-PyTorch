import torch
import torch.nn as nn

from typing import List


class BasicBlock(nn.Module):
    """basic block of ResNet, with optional ViBERTgrid early fusion

    Parameters
    ----------
    in_channel : int
        number of input channel
    out_channel : int
        number of output channel
    downsample : bool, optional
        apply downsampling to the given feature map, by default False
    grid_channel : int, optional
        number of ViBERTgrid channel, apply early fusion if given, by default None
    """

    def __init__(
        self, in_channel: int,
        out_channel: int,
        downsample: bool = False,
        grid_channel: int = None
    ) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.grid_channel = grid_channel
        if downsample:
            self.conv_1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                    kernel_size=3, stride=2, padding=1, bias=False)
            self.conv_shortcut = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                           kernel_size=1, stride=2, padding=0, bias=False)
        else:
            self.conv_1 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                    kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_shortcut = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                           kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_1 = nn.BatchNorm2d(out_channel)
        self.relu_1 = nn.ReLU(inplace=True)

        if self.grid_channel is not None:
            self.conv_1_1 = nn.Conv2d(in_channels=out_channel+grid_channel, out_channel=out_channel,
                                      kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(out_channel)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, grid: torch.Tensor = None):
        x_m = self.conv_1(x)
        x_m = self.bn_1(x_m)
        x_m = self.relu_1(x_m)

        if grid is not None and self.grid_channel is not None:
            x_g = torch.concat([x_m, grid], dim=1)
            x_m = self.conv_1_1(x_g)

        x_m = self.conv_2(x_m)
        x_m = self.bn_2(x_m)

        x_c = self.conv_shortcut(x)

        return self.relu_2(x_m + x_c)


class ResNetFPN_ViBERTgrid(nn.Module):
    """ResNetFPN with ViBERTgrid early fusion

    Parameters
    ----------
    block : nn.Module
        block used in ResNet
    size_list : List
        List of number of blocks in ResNet
    grid_channel : int
        number of ViBERTgrid channels
    pyramid_channel : int, optional
        number of feature channels in FPN, by default 256
    fuse_channel : int, optional
        number of channels of P_{fuse} mentioned in sec 3.1.2 of the paper, by default 256
    """
    def __init__(
        self,
        block: nn.Module,
        size_list: List,
        grid_channel: int,
        pyramid_channel: int = 256,
        fuse_channel: int = 256
    ) -> None:
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_2_x = self._make_ResNet_layer(block, in_channel=64, out_channel=64,
                                                block_num=size_list[0], downsample=False)
        self.conv_3_x = self._make_ResNet_layer(block, in_channel=64, out_channel=128,
                                                block_num=size_list[1], downsample=True, grid_channel=grid_channel)
        self.conv_4_x = self._make_ResNet_layer(block, in_channel=128, out_channel=256,
                                                block_num=size_list[2], downsample=True)
        self.conv_5_x = self._make_ResNet_layer(block, in_channel=256, out_channel=512,
                                                block_num=size_list[3], downsample=True)

        self.skip_1 = nn.Conv2d(in_channels=64, out_channels=pyramid_channel,
                                kernel_size=1, stride=1, padding=0, bias=False)
        self.upsample_1 = nn.Conv2d(in_channels=512, out_channels=pyramid_channel,
                                    kernel_size=1, stride=1, padding=0, bias=False)
        self.merge_1 = nn.Conv2d(in_channels=pyramid_channel, out_channels=pyramid_channel,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.skip_2 = nn.Conv2d(in_channels=128, out_channels=pyramid_channel,
                                kernel_size=1, stride=1, padding=0, bias=False)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_2 = nn.Conv2d(in_channels=pyramid_channel, out_channels=pyramid_channel,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.skip_3 = nn.Conv2d(in_channels=64, out_channels=pyramid_channel,
                                kernel_size=1, stride=1, padding=0, bias=False)
        self.upsample_3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_3 = nn.Conv2d(in_channels=pyramid_channel, out_channels=pyramid_channel,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.upsample_4 = nn.Upsample(scale_factor=2, mode='nearest')

        self.fuse_up_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse_up_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse_up_3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.fuse = nn.Conv2d(in_channels=4 * pyramid_channel, out_channels=fuse_channel,
                              kernel_size=1, stride=1, padding=0, bias=False)

    def _make_ResNet_layer(self, block, in_channel, out_channel, block_num, downsample=True, grid_channel=None):
        layers = []
        for i in range(block_num):
            if i == 0:
                layers.append(block(in_channel, out_channel,
                              downsample=downsample, grid_channel=grid_channel))
            else:
                layers.append(block(in_channel, out_channel, downsample=False))

        return nn.Sequential(*layers)

    def forward(self, input, grid):
        x_1 = self.conv_1(input)
        x_1 = self.pool_1(x_1)
        x_1 = self.conv_2_x(x_1)

        x_2 = self.conv_3_x(x_1, grid)

        x_3 = self.conv_4_x(x_2)

        x_4 = self.conv_5_x(x_3)

        x_5_1 = self.upsample_1(x_4)
        x_5_2 = self.skip_1(x_3)
        x_5 = self.fuse_up_1(x_5_1 + x_5_2)

        x_6_1 = self.upsample_2(x_5)
        x_6_2 = self.skip_2(x_2)
        x_6 = self.fuse_up_2(x_6_1 + x_6_2)

        x_7_1 = self.upsample_2(x_6)
        x_7_2 = self.skip_2(x_1)
        x_7 = self.fuse_up_2(x_7_1 + x_7_2)

        x_fuse_1 = self.fuse_up_1(x_5_1)
        x_fuse_2 = self.fuse_up_2(x_6_1)
        x_fuse_3 = self.fuse_up_3(x_7_1)
        x_fuse = torch.concat([x_fuse_1, x_fuse_2, x_fuse_3, x_7], dim=1)
        x_fuse = self.fuse(x_fuse)

        return x_fuse
