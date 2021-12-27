import torch
import torch.nn as nn

from typing import Tuple, Any


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
        self.activation_1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
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
        self.linear = nn.Linear(in_features=in_channels,
                                out_features=out_channels, bias=bias)

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
            raise TypeError(
                f'roi_shape must be int or Tuple, {type(roi_shape)} given')

        self.ROI_embedding_net = ROIEmbedding(
            num_channels=roi_channel,
            roi_shape=(ROI_output[0], ROI_output[1])
        )

        self.fuse_embedding_net = SingleLayer(
            in_channels=self.BERT_dimension + 1024,
            out_channels=1024,
            bias=True
        )

    def forward(self, ROI_output: torch.Tensor, BERT_embeddings: torch.Tensor):
        """forward propagation of late fusion

        Parameters
        ----------
        ROI_output : torch.Tensor
            ROIs obtained from grid_roi_align
        BERT_embeddings : torch.Tensor
            BERT embeddings obtained from BERTgrid_generator

        Returns
        -------
        fuse_embeddings : torch.Tensor
            fused features
        """
        _, _, BERT_dimension = BERT_embeddings.shape

        # (bs*seq_len, C, ROI_H, ROI_W) -> (bs*seq_len, 1024)
        ROI_embeddings: torch.Tensor = self.ROI_embedding_net(ROI_output)
        # (bs*seq_len, 1024) + (bs, seq_len, BERT_dimension) -> (bs*seq_len)
        fuse_embeddings = torch.cat(
            (ROI_embeddings, BERT_embeddings.reshape(-1, self.BERT_dimension)), dim=1)

        # (bs*seq_len, 1024)
        fuse_embeddings = self.fuse_embedding_net(fuse_embeddings)

        return fuse_embeddings


class FieldTypeClassification(nn.Module):
    """a simplified version of field type classification,  
    discard the original two-stage classification pipeline

    apply classification to all ROIs seperately

    Parameters
    ----------
    num_classes : [type]
        [description]
    fuse_embedding_shape : [type]
        [description]
    loss_weights : [type], optional
        [description], by default None

    """

    def __init__(
        self,
        num_classes: int,
        fuse_embedding_channel: int,
        loss_weights=None
    ) -> None:
        super().__init__()
        self.classification_net = SingleLayer(
            in_channels=fuse_embedding_channel,
            out_channels=num_classes,
            bias=True
        )
        if loss_weights is not None:
            self.field_type_classification_loss = nn.CrossEntropyLoss(
                weight=loss_weights)
        else:
            self.field_type_classification_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        fuse_embeddings: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        class_labels: torch.Tensor
    ) -> torch.Tensor:
        """a simplified version of field type classification,  
        discard the original two-stage classification pipeline

        apply classification to all ROIs seperately

        Parameters
        ----------
        fuse_embeddings : torch.Tensor
            late fusion results from late_fusion
        coords : torch.Tensor
            coords from SROIEDataset
        mask : torch.Tensor
            mask from SROIEDataset
        class_labels : torch.Tensor
            class labels from SROIEDataset

        Returns
        -------
        field_type_classification_loss : torch.Tensor
            classification loss
        """
        device = coords.device

        bs = coords.shape[0]
        seq_len = coords.shape[1]
        field_types = class_labels.shape[1]
        # (bs*seq_len, 1024) -> (bs, seq_len, 1024)
        fuse_embeddings = fuse_embeddings.reshape(
            bs, -1, fuse_embeddings.shape[-1])

        # (bs*seq_len, field_types)
        pred_class_orig = self.classification_net(
            fuse_embeddings.reshape(-1, fuse_embeddings.shape[-1]))
        pred_class_orig = pred_class_orig.reshape(bs, seq_len, field_types)

        # TODO Low efficiency implementation, need optimization
        classification_loss_val = 0
        label_class = []
        pred_class = []
        for bs_index in range(bs):
            for seq_index in range(seq_len):
                if mask[bs_index, seq_index] == 1:
                    cur_coor = coords[bs_index, seq_index, :]
                    if cur_coor[1] == cur_coor[3]:
                        cur_coor[3] += 1
                    if cur_coor[0] == cur_coor[2]:
                        cur_coor[2] += 1
                    curr_label_class = class_labels[bs_index, :, cur_coor[1]:cur_coor[3],
                                                    cur_coor[0]:cur_coor[2]]
                    curr_label_class = curr_label_class.argmax(
                        dim=0).reshape(-1)
                    curr_label_class = curr_label_class.bincount().argmax().item()
                    label_class.append(curr_label_class)
                    pred_class.append(
                        pred_class_orig[None, bs_index, seq_index])
                else:
                    continue

        label_class = torch.tensor(label_class).to(device)
        pred_class = torch.cat(pred_class, dim=0).to(device)

        # TODO computing CELoss is time-comsuming
        classification_loss_val = self.field_type_classification_loss(
            pred_class, label_class)
        return classification_loss_val
