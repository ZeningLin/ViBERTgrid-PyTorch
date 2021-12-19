import torch
import torch.nn as nn


class SemanticSegmentationEncoder(nn.Module):
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


class SemanticSegmentationClassifier(nn.Module):
    """auxiliary semantic segmentation head,  
       apply two multi-class classification to the feature map

    Parameters
    ----------
    p_fuse_channel : int
        [description]
    loss_weights : torch.Tensor, optional
        [description], by default None
    """

    def __init__(self, p_fuse_channel: int, loss_weights: torch.Tensor = None) -> None:
        super().__init__()
        self.semantic_segmentation_encoder = SemanticSegmentationEncoder(
            fuse_channel=p_fuse_channel
        )

        if loss_weights is not None:
            self.aux_loss_1 = nn.CrossEntropyLoss(weight=loss_weights)
            self.aux_loss_2 = nn.CrossEntropyLoss(weight=loss_weights)
        else:
            self.aux_loss_1 = nn.CrossEntropyLoss()
            self.aux_loss_2 = nn.CrossEntropyLoss()

    def forward(
        self,
        fuse_feature: torch.Tensor,
        pos_neg_labels: torch.Tensor,
        class_labels: torch.Tensor
    ) -> torch.Tensor:
        """forward propagation of SemanticSegmentationClassifier

        Parameters
        ----------
        fuse_feature : torch.Tensor
            p_fuse feature maps mentioned in sec 3.1.2 of the paper
        pos_neg_labels : torch.Tensor
            pos_neg labels from SROIEDataset
        class_labels : torch.Tensor
            class labels from SROIEDataset

        Returns
        -------
        aux_loss : torch.Tensor
            auxiliary segmentation loss
        """

        x_out_1, x_out_2 = self.semantic_segmentation_encoder(fuse_feature)

        aux_loss_1_val = self.aux_loss_1(x_out_1, pos_neg_labels.argmax(dim=1))
        aux_loss_2_val = self.aux_loss_2(x_out_2, class_labels.argmax(dim=1))

        return aux_loss_1_val + aux_loss_2_val
