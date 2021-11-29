import torch.nn as nn


class SemanticSegmentationNet(nn.Module):
    def __init__(self, fuse_channel: int) -> None:
        """semantic segmentation net
           two 3*3 conv + upsample + 1*1 conv

        Parameters
        ----------
        fuse_channel : int
            number of channels in p_fuse
        """
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=fuse_channel,
            out_channels=fuse_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn_1 = nn.BatchNorm2d(num_features=fuse_channel)
        self.activation_1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(
            in_channels=fuse_channel,
            out_channels=fuse_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn_2 = nn.BatchNorm2d(num_features=fuse_channel)
        self.activation_2 = nn.ReLU(inplace=True)
        # TODO: what kind of upsampling should be used?
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=4)
        self.conv_3_1 = nn.Conv2d(
            in_channels=fuse_channel,
            out_channels=3,
            kernel_size=1
        )
        self.conv_3_2 = nn.Conv2d(
            in_channels=fuse_channel,
            out_channels=fuse_channel,
            kernel_size=1
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.activation_2(x)
        x = self.upsampling(x)

        x_1 = self.conv_3_1(x)
        x_2 = self.conv_3_2(x)

        return x_1, x_2
