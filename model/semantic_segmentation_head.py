import torch
import torch.nn as nn

from pipeline.custom_loss import CrossEntropyLossRandomSample, CrossEntropyLossOHEM
from typing import List, Tuple


class SemanticSegmentationEncoder(nn.Module):
    """semantic segmentation net
       two 3*3 conv + upsample + 1*1 conv

    Parameters
    ----------
    fuse_channel : int
        number of channels in p_fuse
    num_classes : int
        number of classes
    """

    def __init__(self, fuse_channel: int, num_classes: int) -> None:
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=fuse_channel,
            out_channels=fuse_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn_1 = nn.BatchNorm2d(num_features=fuse_channel)
        self.activation_1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(
            in_channels=fuse_channel,
            out_channels=fuse_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn_2 = nn.BatchNorm2d(num_features=fuse_channel)
        self.activation_2 = nn.ReLU(inplace=True)
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=4)
        self.conv_3_1 = nn.Conv2d(
            in_channels=fuse_channel, out_channels=3, kernel_size=1
        )
        self.conv_3_2 = nn.Conv2d(
            in_channels=fuse_channel, out_channels=num_classes, kernel_size=1
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
    """a simplified version of auxiliary semantic segmentation head,
       apply two multi-class classification to the feature map

    Parameters
    ----------
    p_fuse_channel : int
        number of channels in feature map p_fuse
    num_classes: int
        number of classes, background included
    loss_weights : torch.Tensor, optional
        weight tensor used in CrossEntropyLoss, by default None
    loss_1_sample_list: List
        list of numbers of samples for hard example mining in `L_\{AUX-1\}`, by default None
    num_hard_positive: int
        number of hard positive samples for OHEM in `L_\{AUX-2\}`, by default -1
    num_hard_negative: int
        number of hard negative samples for OHEM in `L_\{AUX-2\}`, by default -1
    """

    def __init__(
        self,
        p_fuse_channel: int,
        num_classes: int,
        loss_weights: torch.Tensor = None,
        loss_1_sample_list: List = None,
        num_hard_positive: int = -1,
        num_hard_negative: int = -1,
    ) -> None:
        super().__init__()
        self.semantic_segmentation_encoder = SemanticSegmentationEncoder(
            fuse_channel=p_fuse_channel, num_classes=num_classes
        )

        if loss_weights is not None:
            self.aux_loss_1 = CrossEntropyLossRandomSample(
                sample_list=loss_1_sample_list
            )
            self.aux_loss_2 = CrossEntropyLossOHEM(
                num_hard_positive=num_hard_positive,
                num_hard_negative=num_hard_negative,
                weight=loss_weights,
            )
        else:
            self.aux_loss_1 = CrossEntropyLossRandomSample(
                sample_list=loss_1_sample_list
            )
            self.aux_loss_2 = CrossEntropyLossOHEM(
                num_hard_positive=num_hard_positive,
                num_hard_negative=num_hard_negative,
            )

    def forward(
        self,
        fuse_feature: torch.Tensor,
        seg_indices: Tuple[torch.Tensor],
        seg_classes: Tuple[torch.Tensor],
        coors: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        """forward propagation of SemanticSegmentationClassifier

        Parameters
        ----------
        fuse_feature : torch.Tensor
            p_fuse feature maps mentioned in sec 3.1.2 of the paper
        seg_indices: Tuple[torch.Tensor]
        seg_classes: Tuple[torch.Tensor]
        coors: torch.Tensor

        Returns
        -------
        aux_loss : torch.Tensor
            auxiliary segmentation loss
        pred_mask : torch.Tensor
            prediction of semantic segmentation mask
        pred_ss : torch.Tensor
            prediction of semantic segmentation class
        """
        device = fuse_feature.device

        x_out_1: torch.Tensor
        x_out_2: torch.Tensor
        x_out_1, x_out_2 = self.semantic_segmentation_encoder(fuse_feature)

        batch_size = x_out_1.shape[0]
        feat_shape = x_out_1.shape[-2:]
        pos_neg_labels = torch.zeros(
            (batch_size, feat_shape[0], feat_shape[1]), dtype=torch.long, device=device
        )
        class_labels = torch.zeros(
            (batch_size, feat_shape[0], feat_shape[1]), dtype=torch.long, device=device
        )
        for batch_index in range(batch_size):
            prev_segment = -1
            for token_index in range(seg_classes[batch_index].shape[0]):
                curr_segment = seg_indices[batch_index][token_index].item()
                if curr_segment == prev_segment:
                    prev_segment = curr_segment
                    continue
                prev_segment = curr_segment

                curr_class = seg_classes[batch_index][token_index]
                curr_coors = coors[batch_index][token_index]
                pos_neg_labels[
                    batch_index,
                    curr_coors[1] : curr_coors[3],
                    curr_coors[0] : curr_coors[2],
                ] = (
                    1 if curr_class > 0 else 2
                )
                class_labels[
                    batch_index,
                    curr_coors[1] : curr_coors[3],
                    curr_coors[0] : curr_coors[2],
                ] = curr_class

        aux_loss_1_val = self.aux_loss_1(x_out_1, pos_neg_labels)
        aux_loss_2_val = self.aux_loss_2(
            x_out_2,
            class_labels,
        )

        del class_labels
        del pos_neg_labels

        return aux_loss_1_val + aux_loss_2_val, x_out_1, x_out_2
