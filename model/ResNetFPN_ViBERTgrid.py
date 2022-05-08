import torch
import torch.nn as nn

from torchvision.models import resnet18, resnet34

from typing import List


class BasicBlock_old(nn.Module):
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
        self,
        in_channel: int,
        out_channel: int,
        downsample: bool = False,
        grid_channel: int = None,
    ) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.grid_channel = grid_channel
        if downsample:
            self.conv_1 = nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
            self.conv_shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_channel),
            )
        else:
            self.conv_1 = nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.conv_shortcut = nn.Identity()
        self.bn_1 = nn.BatchNorm2d(out_channel)
        self.relu_1 = nn.ReLU(inplace=True)

        if self.grid_channel is not None:
            self.conv_1_1 = nn.Conv2d(
                in_channels=out_channel + grid_channel,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )

        self.conv_2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
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
        self,
        in_channel: int,
        out_channel: int,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        if downsample:
            self.conv_1 = nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
            self.conv_shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_channel),
            )
        else:
            self.conv_1 = nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.conv_shortcut = nn.Identity()
        self.bn_1 = nn.BatchNorm2d(out_channel)
        self.relu_1 = nn.ReLU(inplace=True)

        self.conv_2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn_2 = nn.BatchNorm2d(out_channel)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        x_m = self.conv_1(x)
        x_m = self.bn_1(x_m)
        x_m = self.relu_1(x_m)

        x_m = self.conv_2(x_m)
        x_m = self.bn_2(x_m)

        x_c = self.conv_shortcut(x)

        return self.relu_2(x_m + x_c)


class DBlock(nn.Module):
    """Adaption of ResNet basic block, with optional ViBERTgrid early fusion.
    Referring to
    *He et al. Bag of Tricks for Image Classification with Convolutional Neural Networks. CVPR, 2019*

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
        self,
        in_channel: int,
        out_channel: int,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        if downsample:
            self.conv_1 = nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
            self.conv_shortcut = nn.Sequential(
                # Use AvgPool for down samplilng rather than Conv2d
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_channel),
            )
        else:
            self.conv_1 = nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.conv_shortcut = nn.Identity()
        self.bn_1 = nn.BatchNorm2d(out_channel)
        self.relu_1 = nn.ReLU(inplace=True)

        self.conv_2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn_2 = nn.BatchNorm2d(out_channel)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        x_m = self.conv_1(x)
        x_m = self.bn_1(x_m)
        x_m = self.relu_1(x_m)

        x_m = self.conv_2(x_m)
        x_m = self.bn_2(x_m)

        x_c = self.conv_shortcut(x)

        return self.relu_2(x_m + x_c)


class EarlyFusionLayer(nn.Module):
    """An adaption of normal ResNet layer.
    The first block takes the feature map and BERTgrid as input,
    then apply early fusion. The following layers keep the same
    as the normal ResNet layers

    Parameters
    ----------
    block : nn.Module
        basic block type used in ResNet
    in_channel : int
        number of input channel of the layer
    out_channel : int
        number of output channel of the layer
    block_num : int
        number of blocks inside the layer
    grid_channel : int
        number of ViBERTgrid channel
    downsample : bool, optional
        apply downsampling to the given feature map, by default False, by default True
    """

    def __init__(
        self,
        block,
        in_channel: int,
        out_channel: int,
        block_num: int,
        grid_channel: int,
        downsample=True,
    ) -> None:
        super().__init__()
        self.block_1 = block(in_channel, out_channel, downsample=downsample)
        self.early_fusion = nn.Conv2d(
            in_channels=(out_channel + grid_channel),
            out_channels=out_channel,
            kernel_size=1,
        )
        layers = []
        for _ in range(block_num - 1):
            layers.append(block(in_channel, out_channel, downsample=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, grid):
        x = self.block_1(x)
        early_fuse = torch.cat((x, grid), dim=1)
        early_fuse = self.early_fusion(early_fuse)
        output = self.layers(early_fuse)

        return output


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
        fuse_channel: int = 256,
    ) -> None:
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
        )
        self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv_2_x = self._make_ResNet_layer(
            block,
            in_channel=64,
            out_channel=64,
            block_num=size_list[0],
            downsample=False,
        )
        self.conv_3_x = EarlyFusionLayer(
            block,
            in_channel=64,
            out_channel=128,
            block_num=size_list[1],
            grid_channel=grid_channel,
            downsample=True,
        )
        self.conv_4_x = self._make_ResNet_layer(
            block,
            in_channel=128,
            out_channel=256,
            block_num=size_list[2],
            downsample=True,
        )
        self.conv_5_x = self._make_ResNet_layer(
            block,
            in_channel=256,
            out_channel=512,
            block_num=size_list[3],
            downsample=True,
        )
        self.conv_6_x = nn.Conv2d(
            in_channels=512,
            out_channels=pyramid_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.skip_1 = nn.Conv2d(
            in_channels=256,
            out_channels=pyramid_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.upsample_1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.merge_1 = nn.Conv2d(
            in_channels=pyramid_channel,
            out_channels=pyramid_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.skip_2 = nn.Conv2d(
            in_channels=128,
            out_channels=pyramid_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.upsample_2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.merge_2 = nn.Conv2d(
            in_channels=pyramid_channel,
            out_channels=pyramid_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.skip_3 = nn.Conv2d(
            in_channels=64,
            out_channels=pyramid_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.upsample_3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.merge_3 = nn.Conv2d(
            in_channels=pyramid_channel,
            out_channels=pyramid_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.upsample_4 = nn.Upsample(scale_factor=2, mode="nearest")

        self.fuse_up_1 = nn.Upsample(scale_factor=8, mode="nearest")
        self.fuse_up_2 = nn.Upsample(scale_factor=4, mode="nearest")
        self.fuse_up_3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse = nn.Conv2d(
            in_channels=4 * pyramid_channel,
            out_channels=fuse_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def _make_ResNet_layer(
        self, block, in_channel, out_channel, block_num, downsample=True
    ):
        layers = []
        for i in range(block_num):
            if i == 0:
                layers.append(block(in_channel, out_channel, downsample=downsample))
            else:
                layers.append(block(out_channel, out_channel, downsample=False))

        return nn.Sequential(*layers)

    def forward(self, input, grid):
        x_1 = self.conv_1(input)
        x_1 = self.pool_1(x_1)
        x_1 = self.conv_2_x(x_1)

        x_2 = self.conv_3_x(x_1, grid)

        x_3 = self.conv_4_x(x_2)

        x_4 = self.conv_5_x(x_3)
        x_4 = self.conv_6_x(x_4)

        x_5_1 = self.upsample_1(x_4)
        x_5_2 = self.skip_1(x_3)
        x_5 = self.merge_1(x_5_1 + x_5_2)

        x_6_1 = self.upsample_2(x_5)
        x_6_2 = self.skip_2(x_2)
        x_6 = self.merge_2(x_6_1 + x_6_2)

        x_7_1 = self.upsample_3(x_6)
        x_7_2 = self.skip_3(x_1)
        x_7 = self.merge_3(x_7_1 + x_7_2)

        x_fuse_1 = self.fuse_up_1(x_4)
        x_fuse_2 = self.fuse_up_2(x_5)
        x_fuse_3 = self.fuse_up_3(x_6)
        x_fuse = torch.concat([x_fuse_1, x_fuse_2, x_fuse_3, x_7], dim=1)
        x_fuse = self.fuse(x_fuse)

        return x_fuse


class ResNetFPN_ViBERTgrid_Pretrained(nn.Module):
    def __init__(
        self,
        resnet_type: str,
        grid_channel: int,
        pyramid_channel: int = 256,
        fuse_channel: int = 256,
    ) -> None:
        super().__init__()
        if resnet_type == "resnet18":
            self.resnet = resnet18(pretrained=True)
            self.norm_fuse_channel = 128
        elif resnet_type == "resnet34":
            self.resnet = resnet34(pretrained=True)
            self.norm_fuse_channel = 128
        else:
            raise ValueError(f"invalid value of resnet_type")

        self.early_fusion = nn.Conv2d(
            in_channels=(grid_channel + self.norm_fuse_channel),
            out_channels=self.norm_fuse_channel,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.num_block_ly2 = len(self.resnet.layer2)

        self.conv_6_x = nn.Conv2d(
            in_channels=512,
            out_channels=pyramid_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.skip_1 = nn.Conv2d(
            in_channels=256,
            out_channels=pyramid_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.upsample_1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.merge_1 = nn.Conv2d(
            in_channels=pyramid_channel,
            out_channels=pyramid_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.skip_2 = nn.Conv2d(
            in_channels=128,
            out_channels=pyramid_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.upsample_2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.merge_2 = nn.Conv2d(
            in_channels=pyramid_channel,
            out_channels=pyramid_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.skip_3 = nn.Conv2d(
            in_channels=64,
            out_channels=pyramid_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.upsample_3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.merge_3 = nn.Conv2d(
            in_channels=pyramid_channel,
            out_channels=pyramid_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.upsample_4 = nn.Upsample(scale_factor=2, mode="nearest")

        self.fuse_up_1 = nn.Upsample(scale_factor=8, mode="nearest")
        self.fuse_up_2 = nn.Upsample(scale_factor=4, mode="nearest")
        self.fuse_up_3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.fuse = nn.Conv2d(
            in_channels=4 * pyramid_channel,
            out_channels=fuse_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, input: torch.Tensor, BERTgrid: torch.Tensor):
        x_1 = self.resnet.conv1(input)
        x_1 = self.resnet.bn1(x_1)
        x_1 = self.resnet.relu(x_1)
        x_1 = self.resnet.maxpool(x_1)
        x_1 = self.resnet.layer1(x_1)

        x_2 = self.resnet.layer2[0](x_1)
        x_concat = torch.cat((x_2, BERTgrid), dim=1)
        x_2 = self.early_fusion(x_concat)
        for i in range(1, self.num_block_ly2):
            x_2 = self.resnet.layer2[i](x_2)

        x_3 = self.resnet.layer3(x_2)

        x_4 = self.resnet.layer4(x_3)
        x_4 = self.conv_6_x(x_4)

        x_5_1 = self.upsample_1(x_4)
        x_5_2 = self.skip_1(x_3)
        x_5 = self.merge_1(x_5_1 + x_5_2)

        x_6_1 = self.upsample_2(x_5)
        x_6_2 = self.skip_2(x_2)
        x_6 = self.merge_2(x_6_1 + x_6_2)

        x_7_1 = self.upsample_3(x_6)
        x_7_2 = self.skip_3(x_1)
        x_7 = self.merge_3(x_7_1 + x_7_2)

        x_fuse_1 = self.fuse_up_1(x_4)
        x_fuse_2 = self.fuse_up_2(x_5)
        x_fuse_3 = self.fuse_up_3(x_6)
        x_fuse = torch.concat([x_fuse_1, x_fuse_2, x_fuse_3, x_7], dim=1)
        x_fuse = self.fuse(x_fuse)

        return x_fuse


def resnet_18_fpn(grid_channel: int, pretrained: bool = False) -> nn.Module:
    """return ResNet_18_FPN

    Parameters
    ----------
    grid_channel : int
        number of channels in ViBERTgrid

    Returns
    -------
    resnet_18_fpn: nn.Module
        network

    """
    if pretrained:
        return ResNetFPN_ViBERTgrid_Pretrained(
            resnet_type="resnet18", grid_channel=grid_channel
        )
    else:
        block = BasicBlock
        net = ResNetFPN_ViBERTgrid(
            block=block, size_list=[2, 2, 2, 2], grid_channel=grid_channel
        )

        return net


def resnet_34_fpn(grid_channel: int, pretrained: bool = False) -> nn.Module:
    if pretrained:
        return ResNetFPN_ViBERTgrid_Pretrained(
            resnet_type="resnet34", grid_channel=grid_channel
        )
    else:
        block = BasicBlock
        net = ResNetFPN_ViBERTgrid(
            block=block, size_list=[3, 4, 6, 3], grid_channel=grid_channel
        )

        return net


def resnet_18_D_fpn(grid_channel: int) -> nn.Module:
    """return ResNet_18_D_FPN

    Parameters
    ----------
    grid_channel : int
        number of channels in ViBERTgrid

    Returns
    -------
    resnet_18_fpn: nn.Module
        network

    """
    block = DBlock
    net = ResNetFPN_ViBERTgrid(
        block=block, size_list=[2, 2, 2, 2], grid_channel=grid_channel
    )

    return net


def resnet_34_D_fpn(grid_channel: int) -> nn.Module:
    block = DBlock
    net = ResNetFPN_ViBERTgrid(
        block=block, size_list=[3, 4, 6, 3], grid_channel=grid_channel
    )

    return net
