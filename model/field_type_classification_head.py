import torch
import torch.nn as nn

from model.crf import CRF, START_TAG, STOP_TAG
from pipeline.custom_loss import (
    CrossEntropyLossOHEM,
    CrossEntropyLossRandomSample,
    BCELossOHEM,
    BCELossRandomSample,
)

from typing import Optional, Tuple, List, Any, Dict


class AttrProxy(object):
    """Translates index lookups into attribute lookups."""

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class ROIEmbedding(nn.Module):
    """roi embedding modules, convert rois to fused features

    Parameters
    ----------
    num_channels : int
        number of ROI channels
    roi_shape : Any
        shape of ROIs, can be `Tuple` with two elements or `int`

    """

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
    def __init__(self, in_features, out_features, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )

    def forward(self, x):
        return self.linear(x)


class MultipleLayer(nn.Module):
    def __init__(self, in_features, out_features, bias: bool = True) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(
            in_features=in_features, out_features=in_features // 2, bias=bias
        )
        self.nonlinear = nn.ReLU(inplace=True)
        self.linear_2 = nn.Linear(
            in_features=in_features // 2,
            out_features=out_features,
            bias=bias,
        )

    def forward(self, x):
        x = self.linear_1(x)
        x = self.nonlinear(x)
        x = self.linear_2(x)

        return x


class BinaryClassifier(nn.Module):
    def __init__(
        self, in_channels, bias: bool = True, layer_mode: str = "multi"
    ) -> None:
        super().__init__()
        assert layer_mode in [
            "single",
            "multi",
        ], f"layer_mode must be single or multi, {layer_mode} given"

        if layer_mode == "single":
            self.layer = SingleLayer(in_features=in_channels, out_features=1, bias=bias)
        else:
            self.layer = MultipleLayer(in_features=in_channels, out_features=1)

    def forward(self, x):
        x = self.layer(x)
        return x


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
            in_features=self.BERT_dimension + 1024, out_features=1024, bias=True
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


class FieldTypeClassification(nn.Module):
    """field type classification, the original design of the paper

    apply classification to all ROIs seperately

    Parameters
    ----------
    num_classes : int
        number of classes
    fuse_embedding_channel : int
        number of channels of fuse embeddings
    loss_weights : torch.Tensor, optional
        weights used in CrossEntropyLoss, deal with data imbalance, by default None
    num_hard_positive_1: int, optional
        number of hard positive samples for OHEM in `L_1`, by default -1
    num_hard_negative_1: int, optional
        number of hard negative samples for OHEM in `L_1`, by default -1
    num_hard_positive_2: int, optional
        number of hard positive samples for OHEM in `L_2`, by default -1
    num_hard_negative_2: int, optional
        number of hard negative samples for OHEM in `L_2`, by default -1
    random: bool, optional
        apply random sampling before OHEM or not
    layer_mode: str, optional
        type of classifier, `single` for a single layer perceptron, `multi` for a MLP
    work_mode: str, optional
        work mode of the model, controls the return values, `train`, `eval` or `inference`

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
        random: bool = False,
        layer_mode: str = "multi",
        work_mode: str = "train",
    ) -> None:
        super().__init__()

        assert work_mode in [
            "train",
            "eval",
            "inference",
        ], f"mode must be 'train' 'eval' or 'inference', {work_mode} given"
        self.work_mode = work_mode

        self.num_classes = num_classes
        self.fuse_embedding_channel = fuse_embedding_channel

        self.pos_neg_classification_net = BinaryClassifier(
            in_channels=fuse_embedding_channel, bias=True, layer_mode=layer_mode
        )

        if self.work_mode == "inference":
            self.pos_neg_classification_loss = None
        else:
            self.pos_neg_classification_loss = BCELossRandomSample(
                sample_list=[num_hard_negative_1, num_hard_positive_1]
            )

        for idx in range(self.num_classes - 1):
            self.add_module(
                f"category_classification_net_{idx}",
                BinaryClassifier(
                    in_channels=fuse_embedding_channel, bias=True, layer_mode=layer_mode
                ),
            )
            if self.work_mode != "inference":
                if loss_weights is not None:
                    self.add_module(
                        f"field_type_classification_loss_{idx}",
                        BCELossOHEM(
                            num_hard_positive=num_hard_positive_2,
                            num_hard_negative=num_hard_negative_2,
                            weight=loss_weights,
                            random=random,
                        ),
                    )
                else:
                    self.add_module(
                        f"field_type_classification_loss_{idx}",
                        BCELossOHEM(
                            num_hard_positive=num_hard_positive_2,
                            num_hard_negative=num_hard_negative_2,
                            random=random,
                        ),
                    )

        self.category_classification_net = AttrProxy(
            self, "category_classification_net_"
        )

        if self.work_mode == "inference":
            self.field_type_classification_loss = None
        else:
            self.field_type_classification_loss = AttrProxy(
                self, "field_type_classification_loss_"
            )

    def inference(
        self,
        fuse_embeddings: torch.Tensor,
    ):
        device = fuse_embeddings.device

        # (bs*seq_len, 1024)
        fuse_embeddings = fuse_embeddings.reshape((-1, self.fuse_embedding_channel))

        # (pure_len, 1)
        pred_pos_neg: torch.Tensor
        pred_pos_neg = self.pos_neg_classification_net(fuse_embeddings).squeeze(1)

        # (pure_len)
        pred_pos_neg_mask = pred_pos_neg.detach().sigmoid().ge(0.5)
        pos_fuse_embeddings = fuse_embeddings[pred_pos_neg_mask]

        class_pred = torch.zeros(
            (fuse_embeddings.shape[0], self.num_classes),
            dtype=pred_pos_neg.dtype,
            device=device,
        )
        class_pred[:, 0] = pred_pos_neg.detach().sigmoid()

        if pos_fuse_embeddings.shape[0] != 0:
            for class_index in range(self.num_classes - 1):
                curr_class_pred: torch.Tensor = self.category_classification_net[
                    class_index
                ](pos_fuse_embeddings).squeeze(1)

                class_pred[:, class_index + 1][
                    pred_pos_neg_mask
                ] = curr_class_pred.detach().sigmoid()

        return class_pred

    def forward(
        self,
        fuse_embeddings: torch.Tensor,
        segment_classes: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        """field type classification,

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

        segment_classes: torch.Tensor = (
            torch.cat(segment_classes, dim=0).to(device).long()
        )
        label_pos_neg = segment_classes > 0

        # (bs*seq_len, 1024)
        fuse_embeddings = fuse_embeddings.reshape((-1, self.fuse_embedding_channel))
        assert fuse_embeddings.shape[0] == segment_classes.shape[0]

        # (pure_len, 1)
        pred_pos_neg: torch.Tensor
        pred_pos_neg = self.pos_neg_classification_net(fuse_embeddings).squeeze(1)
        pos_neg_classification_loss_val = self.pos_neg_classification_loss(
            pred_pos_neg, label_pos_neg.float()
        )
        # (pure_len)
        pred_pos_neg_mask = pred_pos_neg.detach().sigmoid().ge(0.5)
        pos_fuse_embeddings = fuse_embeddings[pred_pos_neg_mask]

        class_pred = torch.zeros(
            (fuse_embeddings.shape[0], self.num_classes),
            dtype=pred_pos_neg.dtype,
            device=device,
        )
        class_pred[:, 0] = pred_pos_neg.detach().sigmoid()
        classification_loss_val = torch.zeros((1,), device=device)

        if pos_fuse_embeddings.shape[0] != 0:
            for class_index in range(self.num_classes - 1):
                curr_class_pred: torch.Tensor = self.category_classification_net[
                    class_index
                ](pos_fuse_embeddings).squeeze(1)
                curr_class_label = segment_classes[pred_pos_neg_mask] == (
                    class_index + 1
                )
                classification_loss_val += self.field_type_classification_loss[
                    class_index
                ](curr_class_pred, curr_class_label.float())

                class_pred[:, class_index + 1][
                    pred_pos_neg_mask
                ] = curr_class_pred.detach().sigmoid()

        return (
            pos_neg_classification_loss_val + classification_loss_val,
            # classification_loss_val,
            segment_classes.int(),
            class_pred,
        )


class SimplifiedFieldTypeClassification(nn.Module):
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
    num_hard_positive_1: int, optional
        number of hard positive samples for OHEM in `L_1`, by default -1
    num_hard_negative_1: int, optional
        number of hard negative samples for OHEM in `L_1`, by default -1
    num_hard_positive_2: int, optional
        number of hard positive samples for OHEM in `L_2`, by default -1
    num_hard_negative_2: int, optional
        number of hard negative samples for OHEM in `L_2`, by default -1
    random: bool, optional
        apply random sampling before OHEM or not
    layer_mode: str, optional
        type of classifier, `single` for a single layer perceptron, `multi` for a MLP
    work_mode: str, optional
        work mode of the model, controls the return values, `train`, `eval` or `inference`
    add_pos_neg: bool, optioanl
        use an additional pos_neg classifier which may boost the recall, by default True

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
        random: bool = False,
        layer_mode: str = "multi",
        work_mode: str = "train",
        add_pos_neg: bool = True,
    ) -> None:
        super().__init__()

        assert work_mode in [
            "train",
            "eval",
            "inference",
        ], f"mode must be 'train' 'eval' or 'inference', {work_mode} given"
        self.work_mode = work_mode

        assert layer_mode in [
            "single",
            "multi",
        ], f"layer_mode must be single or multi, {layer_mode} given"

        self.num_classes = num_classes
        self.fuse_embedding_channel = fuse_embedding_channel

        if layer_mode == "sigle":
            if self.work_mode == "inference":
                self.pos_neg_classification_net = None
            else:
                self.pos_neg_classification_net = SingleLayer(
                    in_features=fuse_embedding_channel, out_features=2, bias=True
                )
            self.category_classification_net = SingleLayer(
                in_features=fuse_embedding_channel, out_features=num_classes, bias=True
            )
        else:
            if self.work_mode == "inference":
                self.pos_neg_classification_net = None
            else:
                self.pos_neg_classification_net = MultipleLayer(
                    in_features=fuse_embedding_channel, out_features=2, bias=True
                )
            self.category_classification_net = MultipleLayer(
                in_features=fuse_embedding_channel,
                out_features=num_classes,
            )

        if self.work_mode == "inference":
            self.pos_neg_classification_loss = None
            self.field_type_classification_loss = None
        else:
            self.pos_neg_classification_loss = CrossEntropyLossOHEM(
                num_hard_positive=num_hard_positive_1,
                num_hard_negative=num_hard_negative_1,
                random=random,
            )
            if loss_weights is not None:
                self.field_type_classification_loss = CrossEntropyLossOHEM(
                    num_hard_positive=num_hard_positive_2,
                    num_hard_negative=num_hard_negative_2,
                    weight=loss_weights,
                    random=random,
                )
            else:
                self.field_type_classification_loss = CrossEntropyLossOHEM(
                    num_hard_positive=num_hard_positive_2,
                    num_hard_negative=num_hard_negative_2,
                    random=random,
                )

        self.add_pos_neg = add_pos_neg

    def inference(self, fuse_embeddings: torch.Tensor):
        # (bs*seq_len)
        fuse_embeddings = fuse_embeddings.reshape((-1, self.fuse_embedding_channel))
        # (pure_len, field_types)
        pred_class: torch.Tensor
        pred_class = self.category_classification_net(fuse_embeddings)

        return pred_class.detach().softmax(dim=1)

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
        segment_classes : Tuple[torch.Tensor]
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

        # (pure_len, field_types)
        pred_class: torch.Tensor
        pred_class = self.category_classification_net(fuse_embeddings)
        classification_loss_val = self.field_type_classification_loss(
            pred_class,
            label_class,
        )

        if self.add_pos_neg:
            return (
                pos_neg_classification_loss_val + classification_loss_val,
                label_class.int(),
                pred_class.detach().softmax(dim=1),
            )
        else:
            return (
                classification_loss_val,
                label_class.int(),
                pred_class.detach().softmax(dim=1),
            )


class CRFFieldTypeClassification(nn.Module):
    def __init__(
        self,
        tag_to_idx: Dict,
        fuse_embedding_channel: int,
        layer_mode: str = "multi",
        work_mode: str = "train",
    ) -> None:
        """field type classification head with CRF layer
            apply multiclass classification

        Parameters
        ----------
        tag_to_idx : Dict
            a dictionary that provides mapping from tag-name to index,
            containing `num_classes` elements
        fuse_embedding_channel : int
            number of channels of fuse embeddings
        layer_mode: str, optional
            type of classifier, `single` for a single layer perceptron, `multi` for a MLP
        work_mode: str, optional
            work mode of the model, controls the return values, `train`, `eval` or `inference`
        
        """
        super().__init__()

        assert work_mode in [
            "train",
            "eval",
            "inference",
        ], f"mode must be 'train' 'eval' or 'inference', {work_mode} given"
        self.work_mode = work_mode

        assert layer_mode in [
            "single",
            "multi",
        ], f"layer_mode must be single or multi, {layer_mode} given"

        self.num_classes = len(tag_to_idx)
        self.num_tags = self.num_classes + 2

        assert (
            max(tag_to_idx.values()) == self.num_classes - 1
        ), f"invalid tag_to_idx format"
        self.tag_to_idx = tag_to_idx
        self.tag_to_idx[START_TAG] = self.num_classes
        self.tag_to_idx[STOP_TAG] = self.num_classes + 1

        self.fuse_embedding_channel = fuse_embedding_channel
        if layer_mode == "single":
            self.category_classification_net = SingleLayer(
                in_features=fuse_embedding_channel,
                out_features=self.num_tags,
                bias=True,
            )
        else:
            self.category_classification_net = MultipleLayer(
                in_features=fuse_embedding_channel,
                out_features=self.num_tags,
                bias=True,
            )

        self.crf_layer = CRF(self.tag_to_idx)

    def inference(self, fuse_embeddings: torch.Tensor):
        device = fuse_embeddings.device

        # (bs*seq_len)
        fuse_embeddings = fuse_embeddings.reshape((-1, self.fuse_embedding_channel))

        pred_class: torch.Tensor
        pred_class = self.category_classification_net(fuse_embeddings)

        _, tag_seq = self.crf_layer.inference(feats=pred_class)
        tag_seq = torch.tensor(tag_seq, device=device).unsqueeze(1)

        return tag_seq.detach().float()

    def forward(
        self,
        fuse_embeddings: torch.Tensor,
        segment_classes: Tuple[torch.Tensor],
    ):
        device = fuse_embeddings.device
        batch_len_list = [b.shape[0] for b in segment_classes]

        label_class = torch.cat(segment_classes, dim=0).to(device)

        # (bs*seq_len)
        fuse_embeddings = fuse_embeddings.reshape((-1, self.fuse_embedding_channel))
        assert fuse_embeddings.shape[0] == label_class.shape[0]

        pred_class: torch.Tensor
        pred_class = self.category_classification_net(fuse_embeddings)

        if self.training:
            score = torch.zeros((1,), device=device)
            start_index = 0
            for batch_len in batch_len_list:
                end_index = start_index + batch_len
                feat = pred_class[start_index:end_index]
                tag = label_class[start_index:end_index]
                score += self.crf_layer(feats=feat, tags=tag)
                start_index = end_index

            return (
                score / len(batch_len_list),
                label_class.int(),
                pred_class.detach().float(),
            )
        else:
            score = torch.zeros((1,), device=device)
            tag_seq_list = list()
            start_index = 0
            for batch_len in batch_len_list:
                end_index = start_index + batch_len
                feat = pred_class[start_index:end_index]
                score_, tag_seq = self.crf_layer.inference(feats=feat)
                score += score_
                tag_seq_list.append(torch.tensor(tag_seq, device=device))
                start_index = end_index

            tag_seq = torch.cat(tag_seq_list, dim=0).unsqueeze(1)
            return (
                score / len(batch_len_list),
                label_class.int(),
                tag_seq.detach().float(),
            )
