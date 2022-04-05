import torch
import torch.nn as nn

from pipeline.custom_loss import CrossEntropyLossOHEM

from typing import Optional, Tuple, List, Any


class ROIEmbedding(nn.Module):
    def __init__(self, num_channels: int, roi_shape: Any) -> None:
        super().__init__()

        if isinstance(roi_shape, Tuple):
            assert (
                len(roi_shape) == 2
            ), f"roi_shape must be int or two-element tuple, {len(roi_shape)} elements were given"
            num_flatten = num_channels * roi_shape[0] * roi_shape[1]
        elif isinstance(roi_shape, int):
            num_flatten = num_channels * roi_shape * roi_shape
        else:
            raise ValueError("roi_shape must be int or two-element tuple")

        self.conv_1 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_1 = nn.BatchNorm2d(num_channels)
        self.activation_1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_2 = nn.BatchNorm2d(num_channels)
        self.activation_2 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(num_flatten, 1024)

    def forward(self, ROI: torch.Tensor) -> torch.Tensor:
        # ROI feature map -> 1024-d feature vector
        ROI_emb = self.conv_1(ROI)
        ROI_emb = self.bn_1(ROI_emb)
        ROI_emb = self.activation_1(ROI_emb)
        ROI_emb = self.conv_2(ROI_emb)
        ROI_emb = self.bn_2(ROI_emb)
        ROI_emb = self.activation_2(ROI_emb)
        ROI_emb = self.flatten(ROI_emb)
        ROI_emb = self.linear(ROI_emb)

        return ROI_emb


class SingleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(
            in_features=in_channels, out_features=out_channels, bias=bias
        )

    def forward(self, x):
        return self.linear(x)


class LateFusion(nn.Module):
    """apply late fusion to ROIs and BERT embeddings

    Parameters
    ----------
    bert_hidden_size : int
        number of channels of bert-model output
    roi_channel : int
        number of channels of roi
    roi_shape : Any
        shape of roi, int or tuple.
        if int, shape = (roi_shape, roi_shape)
    """

    def __init__(self, bert_hidden_size: int, roi_channel: int, roi_shape: Any) -> None:
        super().__init__()

        self.BERT_dimension = bert_hidden_size

        if isinstance(roi_shape, int):
            ROI_output = (roi_shape, roi_shape)
        elif isinstance(roi_shape, Tuple):
            ROI_output = roi_shape
        else:
            raise TypeError(f"roi_shape must be int or Tuple, {type(roi_shape)} given")

        self.ROI_embedding_net = ROIEmbedding(
            num_channels=roi_channel, roi_shape=(ROI_output[0], ROI_output[1])
        )

        self.fuse_embedding_net = SingleLayer(
            in_channels=self.BERT_dimension + 1024, out_channels=1024, bias=True
        )

    def forward(self, ROI_output: torch.Tensor, BERT_embeddings: Tuple[torch.Tensor]):
        """forward propagation of late fusion

        Parameters
        ----------
        ROI_output : torch.Tensor
            ROIs obtained from grid_roi_align
        BERT_embeddings : Tuple[torch.Tensor]
            BERT embeddings obtained from BERTgrid_generator

        Returns
        -------
        fuse_embeddings : torch.Tensor
            fused features
        """
        # (bs*seq_len, C, ROI_H, ROI_W) -> (bs*seq_len, 1024)
        ROI_embeddings: torch.Tensor = self.ROI_embedding_net(ROI_output)
        BERT_embeddings: torch.Tensor = torch.cat(BERT_embeddings, dim=0)
        assert ROI_embeddings.shape[0] == BERT_embeddings.shape[0]

        # (bs*seq_len, 1024) + (bs, seq_len, BERT_dimension) -> (bs*seq_len)
        fuse_embeddings = torch.cat((ROI_embeddings, BERT_embeddings), dim=1)

        # (bs*seq_len, 1024)
        fuse_embeddings = self.fuse_embedding_net(fuse_embeddings)

        return fuse_embeddings


class FieldTypeClassificationSimplified(nn.Module):
    """a simplified version of field type classification,
    discard the original two-stage classification pipeline

    apply classification to all ROIs seperately

    Parameters
    ----------
    num_classes : int
        number of classes
    fuse_embedding_channel : int
        number of channels of fuse embeddings
    loss_weights : torch.Tensor, optional
        weights used in CrossEntropyLoss, deal with data imbalance, by default None
    num_hard_positive: int
        number of hard positive samples for OHEM in `L_2`, by default -1
    num_hard_negative: int
        number of hard negative samples for OHEM in `L_2`, by default -1

    """

    def __init__(
        self,
        num_classes: int,
        fuse_embedding_channel: int,
        loss_weights: Optional[List] = None,
        num_hard_positive_1: int = -1,
        num_hard_negative_1: int = -1,
        num_hard_positive_2: int = -1,
        num_hard_negative_2: int = -1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.fuse_embedding_channel = fuse_embedding_channel
        self.pos_neg_classification_net = SingleLayer(
            in_channels=fuse_embedding_channel, out_channels=2, bias=True
        )
        self.category_classification_net = SingleLayer(
            in_channels=fuse_embedding_channel, out_channels=num_classes, bias=True
        )
        self.pos_neg_classification_loss = CrossEntropyLossOHEM(
        num_hard_positive=num_hard_positive_1, num_hard_negative=num_hard_negative_1
        )
        if loss_weights is not None:
            self.field_type_classification_loss = CrossEntropyLossOHEM(
                num_hard_positive=num_hard_positive_2,
                num_hard_negative=num_hard_negative_2,
                weight=loss_weights,
            )
        else:
            self.field_type_classification_loss = CrossEntropyLossOHEM(
                num_hard_positive=num_hard_positive_2,
                num_hard_negative=num_hard_negative_2,
            )

    def forward(
        self,
        fuse_embeddings: torch.Tensor,
        segment_classes: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        """a simplified version of field type classification,
        discard the original two-stage classification pipeline

        apply classification to all ROIs seperately

        Parameters
        ----------
        fuse_embeddings : torch.Tensor
            late fusion results from late_fusion
        segment_classes: Tuple[torch.Tensor]
            segment_classes from dataset

        Returns
        -------
        field_type_classification_loss : torch.Tensor
            classification loss
        pred_class : torch.Tensor
            prediction class result
        """
        device = fuse_embeddings.device

        label_class = torch.cat(segment_classes, dim=0).to(device).long()
        label_pos_neg = (label_class > 0).long()

        # (bs*seq_len)
        fuse_embeddings = fuse_embeddings.reshape((-1, self.fuse_embedding_channel))
        assert fuse_embeddings.shape[0] == label_class.shape[0]

        # (pure_len, 2)
        pred_pos_neg = self.pos_neg_classification_net(fuse_embeddings)
        pos_neg_classification_loss_val = self.pos_neg_classification_loss(
            pred_pos_neg, label_pos_neg
        )
        # (pure_len)
        pred_pos_neg_mask = torch.argmax(pred_pos_neg.detach(), dim=1)

        # (pure_len, field_types)
        pred_class: torch.Tensor
        pred_class = self.category_classification_net(fuse_embeddings)
        classification_loss_val = self.field_type_classification_loss(
            # pred_class[pred_pos_neg_mask], label_class[pred_pos_neg_mask]
            pred_class,
            label_class,
        )
        # pred_class[~pred_pos_neg_mask] = 0

        return (
            pos_neg_classification_loss_val + classification_loss_val,
            # classification_loss_val,
            label_class.int(),
            pred_class,
        )
