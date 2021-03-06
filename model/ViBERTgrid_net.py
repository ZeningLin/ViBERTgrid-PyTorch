import torch
import torch.nn as nn

from transformers import (
    BertModel,
    BertTokenizer,
    BertConfig,
    RobertaTokenizer,
    RobertaModel,
    RobertaConfig,
    AutoConfig,
)

from typing import Tuple, List, Any, Dict

from model.BERTgrid_generator import BERTgridGenerator
from model.grid_roi_align import GridROIAlign
from model.ResNetFPN_ViBERTgrid import (
    resnet_18_fpn,
    resnet_34_fpn,
    resnet_18_D_fpn,
    resnet_34_D_fpn,
)
from model.field_type_classification_head import (
    FieldTypeClassification,
    SimplifiedFieldTypeClassification,
    CRFFieldTypeClassification,
    LateFusion,
)
from model.semantic_segmentation_head import (
    SemanticSegmentationClassifier,
    SimplifiedSemanticSegmentationClassifier,
)
from pipeline.transform import GeneralizedViBERTgridTransform, ImageList


class ViBERTgridNet(nn.Module):
    """ViBERTgrid network

    Parameters
    ----------
    num_classes
        number of classes in the dataset,
        for example, for SROIE, num_classes=5
    image_mean : float or List[float]
        mean value of each channels of the original image,
        used for input normalization
    image_std : Any
        variance of each channels of the original image,
        used for input normalization
    image_min_size , Tuple[int] or List[int]
        length of the minimum edge after image resize at train mode,
        if tuple or list given, the value will be randomly
        selected from the given values. The original paper
        uses [320, 416, 512, 608, 704].
    image_max_size
        length of the maximum edge after image resize at train mode.
        The original paper uses 800.
    test_image_min_size , optional
        length of the minimum edge after image resize at eval/test mode, by default 512
    bert_model : str, optional
        pretrained BERT model used for BERT embedding, by default 'bert-base-uncased'
    tokenizer : Any, optional
        pretrained BERT tokenizer used for BERT embedding, should be the same as bert_model, by default None
    backbone : str, optional
        backbone used for image feature extraction, by default 'resnet_18_fpn'
    grid_mode : str, optional
        BERTgrid tokens embedding mode.
        Words from the OCR result were splited into several tokens through tokenizer,
        at BERTgrid embedding step, measures shall be taken to
        aggregate these token embeddings back into word-level,
        'mean' mode average token embeddings from the same word,
        'first' mode take the first token embeddings of a word,
        by default 'mean'
    early_fusion_downsampling_ratio , optional
        downsampling ratio of the feature map at early fusion step, by default 8
    roi_shape , optional
        shape of the ROIs after ROIAlign, by default 7
    p_fuse_downsampling_ratio , optional
        downsampling ratio of the P_fuse feature map, by default 4
    roi_align_output_reshape : bool, optional
        controls output reshape at ROIAlign step.
        If True, output shape = (N, seqLen, C, ROI_H, ROI_W)
        else (N * seqLen, C, output_H, output_W),
        by default False.
    late_fusion_fuse_embedding_channel , optional
        number of channels at late_fusion_embedding, by default 1024
    loss_weights : None, List or torch.Tensor, optional
        reduce the impact of data class imbalance, used in CrossEntropyLoss, by default None
    num_hard_positive_main
        number of hard positive samples for OHEM in `L_2`, by default -1
    num_hard_negative_main
        number of hard negative samples for OHEM in `L_2`, by default -1
    loss_aux_sample_list: List
        list of numbers of samples for hard example mining in `L_\{AUX-1\}`, by default None
    num_hard_positive_aux
        number of hard positive samples for OHEM in `L_{AUX-2}`, by default -1
    num_hard_negative_aux
        number of hard negative samples for OHEM in `L_{AUX-2}`, by default -1
    loss_control_lambda : float, optional
        hyperparameters that controls the ratio of auxiliary loss and classification loss, by default 1
    add_pos_neg: bool, optioanl
        use an additional pos_neg classifier in simp mode which may boost the recall, by default True
    classifier_mode: str, optional
        determine which kind of classifier to use.
        "full" refers to the original classifier structure mentioned in the paper,
        using multiple binary classifiers for each field type.
        "simp" refers to the simplified version, using a multi-class classifier for all the
        fields.
        "crf" refers to a field-type classification head with CRF layer.
    tag_to_idx: Dict, optional
        a dictionary that maps the class tag to index, by default None.
        an example for SROIE is shown below:
        `TAG_TO_IDX = {"O": 0, "B-company": 1, "B-date": 2, "B-address": 3, "B-total": 4}`
    ohem_random: bool, optional
        if `True`, apply random sampling and performs OHEM on the sampled loss. the trick may
        boost the performance.
    layer_mode: str, optional
        choose type of classifier used, by default "single".
        "single" use a single layer perceptron for dimension reduction.
        "multi" use a two-layers MLP for dimension reduction.
    work_mode: str, optional
        work mode of the model, can be "train", "eval" or "inference".
        determines the initialization method and return value types of the model.

    """

    def __init__(
        self,
        num_classes,
        image_mean: Any,
        image_std: Any,
        image_min_size: Any,
        image_max_size,
        test_image_min_size=512,
        bert_model: str = "bert-base-uncased",
        tokenizer: Any = None,
        backbone: str = "resnet_18_fpn",
        grid_mode: str = "mean",
        early_fusion_downsampling_ratio=8,
        roi_shape=7,
        p_fuse_downsampling_ratio=4,
        late_fusion_fuse_embedding_channel=1024,
        loss_weights: Any = None,
        num_hard_positive_main_1=-1,
        num_hard_negative_main_1=-1,
        num_hard_positive_main_2=-1,
        num_hard_negative_main_2=-1,
        loss_aux_sample_list: List = None,
        num_hard_positive_aux=-1,
        num_hard_negative_aux=-1,
        loss_control_lambda: float = 1,
        add_pos_neg: bool = True,
        classifier_mode: str = "full",
        tag_to_idx: Dict = None,
        ohem_random: bool = False,
        layer_mode: str = "single",
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
        if tag_to_idx is not None:
            self.num_tokens = len(tag_to_idx)
        else:
            self.num_tokens = num_classes

        # preprocessing stuff
        assert isinstance(
            image_mean, (float, List)
        ), f"image_mean must be float or list of float, {type(image_mean)} given"
        assert isinstance(
            image_std, (float, List)
        ), f"image_std must be float or list of float, {type(image_std)} given"
        if isinstance(image_mean, float):
            image_mean = [image_mean] * 3
        elif len(image_mean) != 3:
            raise ValueError(
                f"image_mean must contain 3 three values, {len(image_mean)} given"
            )
        if isinstance(image_std, float):
            image_std = [image_std] * 3
        elif len(image_std) != 3:
            raise ValueError(
                f"image_std must contain 3 three values, {len(image_std)} given"
            )
        self.image_mean = image_mean
        self.image_std = image_std

        self.test_image_min_size = test_image_min_size
        assert isinstance(
            image_min_size, (int, Tuple, List)
        ), f"image_min_size must be int, Tuple or List, {type(image_min_size)} given"
        image_min_size = list(image_min_size)
        assert isinstance(
            image_max_size, int
        ), f"image_max_size must be int, {type(image_max_size)} given"

        self.image_min_size = image_min_size
        self.image_max_size = image_max_size

        self.transform = GeneralizedViBERTgridTransform(
            image_mean=self.image_mean,
            image_std=self.image_std,
            train_min_size=self.image_min_size,
            test_min_size=self.test_image_min_size,
            max_size=self.image_max_size,
        )

        # bert-model stuff
        self.bert_model_list = {
            "private_bert-base-uncased": 768,
            "bert-base-uncased": 768,
            "bert-base-cased": 768,
            "roberta-base": 768,
            "bert-base-chinese": 768,
            "hfl/chinese-bert-wwm-ext": 768,
            "hfl/chinese-bert-wwm": 768,
        }
        assert (
            bert_model in self.bert_model_list.keys()
        ), f"the given bert model {bert_model} does not exists, see attribute bert_model_list for all bert_models"
        self.bert_hidden_size = self.bert_model_list[bert_model]

        if self.work_mode == "train" or self.work_mode == "inference":
            print("loading pretrained")
            if "roberta-" in bert_model:
                if tokenizer is None:
                    self.tokenizer = RobertaTokenizer.from_pretrained(bert_model)
                elif isinstance(tokenizer, RobertaTokenizer):
                    self.tokenizer = tokenizer
                else:
                    raise ValueError(
                        "invalid value of parameter tokenizer, must be None or callable RobertaTokenizer"
                    )
                self.bert_model = RobertaModel.from_pretrained(bert_model)
            elif "bert-" in bert_model:
                if tokenizer is None:
                    self.tokenizer = BertTokenizer.from_pretrained(bert_model)
                elif isinstance(tokenizer, BertTokenizer):
                    self.tokenizer = tokenizer
                else:
                    raise ValueError(
                        "invalid value of parameter tokenizer, must be None or callable BertTokenizer"
                    )
                self.bert_model = BertModel.from_pretrained(bert_model)
            else:
                raise ValueError("no tokenizer and bert model loaded")
        elif self.work_mode == "eval":
            print("in evaluation mode, no pretrained will be loaded")
            if "bert-" in bert_model:
                if tokenizer is None:
                    self.tokenizer = BertTokenizer.from_pretrained(bert_model)
                elif isinstance(tokenizer, BertTokenizer):
                    self.tokenizer = tokenizer
                else:
                    raise ValueError(
                        "invalid value of parameter tokenizer, must be None or callable BertTokenizer"
                    )
                self.bert_config = AutoConfig.from_pretrained(bert_model)
                self.bert_model = BertModel(self.bert_config)
            elif "roberta-" in bert_model:
                if tokenizer is None:
                    self.tokenizer = RobertaTokenizer.from_pretrained(bert_model)
                elif isinstance(tokenizer, RobertaTokenizer):
                    self.tokenizer = tokenizer
                else:
                    raise ValueError(
                        "invalid value of parameter tokenizer, must be None or callable RobertaTokenizer"
                    )
                self.bert_config = AutoConfig.from_pretrained(bert_model)
                self.bert_model = RobertaModel(self.bert_config)

        # backbone stuff
        self.backbone_list = [
            "resnet_18_fpn",
            "resnet_34_fpn",
            "resnet_18_fpn_pretrained",
            "resnet_34_fpn_pretrained",
            "resnet_18_D_fpn",
            "resnet_34_D_fpn",
        ]
        assert (
            backbone in self.backbone_list
        ), f"the given backbone {backbone} does not exists, see attribute backbone_list for all backbones"
        if backbone == "resnet_18_fpn":
            self.backbone = resnet_18_fpn(grid_channel=self.bert_hidden_size)
            self.p_fuse_channel = 256
        elif backbone == "resnet_34_fpn":
            self.backbone = resnet_34_fpn(grid_channel=self.bert_hidden_size)
            self.p_fuse_channel = 256
        elif backbone == "resnet_18_fpn_pretrained":
            self.backbone = resnet_18_fpn(
                grid_channel=self.bert_hidden_size, pretrained=True
            )
            self.p_fuse_channel = 256
        elif backbone == "resnet_34_fpn_pretrained":
            self.backbone = resnet_34_fpn(
                grid_channel=self.bert_hidden_size, pretrained=True
            )
            self.p_fuse_channel = 256
        elif backbone == "resnet_18_D_fpn":
            self.backbone = resnet_18_D_fpn(grid_channel=self.bert_hidden_size)
            self.p_fuse_channel = 256
        elif backbone == "resnet_34_D_fpn":
            self.backbone = resnet_34_D_fpn(grid_channel=self.bert_hidden_size)
            self.p_fuse_channel = 256
        else:
            raise ValueError("no backbone loaded")

        # bert-grid stuff
        assert grid_mode in [
            "mean",
            "first",
        ], f"grid_mode should be 'mean' or 'first', {grid_mode} were given"
        self.grid_mode = grid_mode
        self.early_fusion_downsampling_ratio = early_fusion_downsampling_ratio

        # grid roi align stuff
        self.roi_shape = roi_shape
        self.p_fuse_downsampling_ratio = p_fuse_downsampling_ratio

        # field-type classification stuff
        self.late_fusion_fuse_embedding_channel = late_fusion_fuse_embedding_channel

        # loss stuff
        if self.work_mode == "inference":
            self.loss_control_lambda = None
        else:
            self.loss_control_lambda = loss_control_lambda

        if loss_weights is None or self.work_mode == "inference":
            self.loss_weights = None
        else:
            if isinstance(loss_weights, List):
                self.loss_weights = torch.tensor(loss_weights)
            elif isinstance(loss_weights, torch.Tensor):
                pass
            else:
                raise TypeError(
                    f"loss_weights must be None, List or torch.Tensor, {type(loss_weights)} given"
                )

        assert classifier_mode in [
            "full",
            "simp",
            "crf",
        ], f"invalid classifier mode, must be 'full', 'simp' or 'crf'"
        self.classifier_mode = classifier_mode

        self.BERTgrid_generator = BERTgridGenerator(
            bert_model=self.bert_model,
            grid_mode=self.grid_mode,
            stride=self.early_fusion_downsampling_ratio,
        )

        self.grid_roi_align_net = GridROIAlign(
            output_size=self.roi_shape,
            step=self.p_fuse_downsampling_ratio,
        )

        self.late_fusion_net = LateFusion(
            bert_hidden_size=self.bert_hidden_size,
            roi_channel=self.p_fuse_channel,
            roi_shape=self.roi_shape,
        )

        if self.classifier_mode == "full":
            if self.work_mode == "inference":
                self.field_type_classification_head = FieldTypeClassification(
                    num_classes=self.num_tokens,
                    fuse_embedding_channel=self.late_fusion_fuse_embedding_channel,
                    layer_mode=layer_mode,
                    work_mode=self.work_mode,
                )
                self.semantic_segmentation_head = None
            else:
                self.field_type_classification_head = FieldTypeClassification(
                    num_classes=self.num_tokens,
                    fuse_embedding_channel=self.late_fusion_fuse_embedding_channel,
                    loss_weights=self.loss_weights,
                    num_hard_positive_1=num_hard_positive_main_1,
                    num_hard_negative_1=num_hard_negative_main_1,
                    num_hard_positive_2=num_hard_positive_main_2,
                    num_hard_negative_2=num_hard_negative_main_2,
                    random=ohem_random,
                    layer_mode=layer_mode,
                    work_mode=self.work_mode,
                )

                self.semantic_segmentation_head = SemanticSegmentationClassifier(
                    p_fuse_channel=self.p_fuse_channel,
                    num_classes=self.num_tokens,
                    loss_weights=self.loss_weights,
                    loss_1_sample_list=loss_aux_sample_list,
                    num_hard_positive=num_hard_positive_aux,
                    num_hard_negative=num_hard_negative_aux,
                )
        elif self.classifier_mode == "simp":
            if self.work_mode == "inference":
                self.field_type_classification_head = SimplifiedFieldTypeClassification(
                    num_classes=self.num_tokens,
                    fuse_embedding_channel=self.late_fusion_fuse_embedding_channel,
                    layer_mode=layer_mode,
                    work_mode=self.work_mode,
                    add_pos_neg=add_pos_neg,
                )
                self.semantic_segmentation_head = None
            else:
                self.field_type_classification_head = SimplifiedFieldTypeClassification(
                    num_classes=self.num_tokens,
                    fuse_embedding_channel=self.late_fusion_fuse_embedding_channel,
                    loss_weights=self.loss_weights,
                    num_hard_positive_1=num_hard_positive_main_1,
                    num_hard_negative_1=num_hard_negative_main_1,
                    num_hard_positive_2=num_hard_positive_main_2,
                    num_hard_negative_2=num_hard_negative_main_2,
                    random=ohem_random,
                    layer_mode=layer_mode,
                    work_mode=self.work_mode,
                )

                self.semantic_segmentation_head = (
                    SimplifiedSemanticSegmentationClassifier(
                        p_fuse_channel=self.p_fuse_channel,
                        num_classes=self.num_tokens,
                        loss_weights=self.loss_weights,
                        loss_1_sample_list=loss_aux_sample_list,
                        num_hard_positive=num_hard_positive_aux,
                        num_hard_negative=num_hard_negative_aux,
                    )
                )

        elif self.classifier_mode == "crf":
            assert tag_to_idx is not None, f"tag_to_idx cannot be None in crf mode"

            self.field_type_classification_head = CRFFieldTypeClassification(
                tag_to_idx=tag_to_idx,
                fuse_embedding_channel=self.late_fusion_fuse_embedding_channel,
                layer_mode=layer_mode,
            )

            if self.work_mode == "inference":
                self.semantic_segmentation_head = None
            else:
                self.semantic_segmentation_head = SemanticSegmentationClassifier(
                    p_fuse_channel=self.p_fuse_channel,
                    num_classes=self.num_tokens,
                    loss_weights=self.loss_weights,
                    loss_1_sample_list=loss_aux_sample_list,
                    num_hard_positive=num_hard_positive_aux,
                    num_hard_negative=num_hard_negative_aux,
                )

    def train(self, mode: bool = True):
        self.work_mode = "train"
        return super().train(mode)

    def eval(self):
        self.work_mode = "eval"
        return super().eval()

    def inference(
        self,
        image: Tuple[torch.Tensor],
        seg_indices: Tuple[torch.Tensor],
        coors: torch.Tensor,
        corpus: torch.Tensor,
        mask: torch.Tensor,
    ):
        image_list: ImageList
        coors: torch.Tensor
        image_list, coors = self.transform(image, coors)
        image_shape = image_list.tensors.shape[-2:]

        # generate BERTgrid
        BERT_embeddings, BERTgrid_embeddings = self.BERTgrid_generator(
            image_shape, seg_indices, corpus, mask, coors
        )

        # encode orig image, early fusion
        p_fuse_features = self.backbone(image_list.tensors, BERTgrid_embeddings)

        # Word-level Field Type Classification Head
        # roi align
        roi_features = self.grid_roi_align_net(p_fuse_features, coors, None)
        # late fusion
        late_fuse_embeddings = self.late_fusion_net(roi_features, BERT_embeddings)
        # field type classification
        pred_label = self.field_type_classification_head.inference(late_fuse_embeddings)

        return pred_label

    def forward(
        self,
        image: Tuple[torch.Tensor],
        seg_indices: Tuple[torch.Tensor],
        segment_classes: Tuple[torch.Tensor],
        coors: torch.Tensor,
        corpus: torch.Tensor,
        mask: torch.Tensor,
    ):
        image_list: ImageList
        coors: torch.Tensor
        image_list, coors = self.transform(image, coors)

        image_shape = image_list.tensors.shape[-2:]

        # generate BERTgrid
        BERT_embeddings, BERTgrid_embeddings = self.BERTgrid_generator(
            image_shape, seg_indices, corpus, mask, coors
        )

        # encode orig image, early fusion
        p_fuse_features = self.backbone(image_list.tensors, BERTgrid_embeddings)

        # Auxiliary Semantic Segmentation Head
        loss_aux, pred_mask, pred_ss = self.semantic_segmentation_head(
            p_fuse_features, segment_classes, coors
        )

        # Word-level Field Type Classification Head
        # roi align
        roi_features = self.grid_roi_align_net(p_fuse_features, coors, None)
        # late fusion
        late_fuse_embeddings = self.late_fusion_net(roi_features, BERT_embeddings)
        # field type classification
        loss_c, gt_label, pred_label = self.field_type_classification_head(
            late_fuse_embeddings, segment_classes
        )

        total_loss = loss_c + self.loss_control_lambda * loss_aux

        if self.work_mode == "train" and self.training:
            return total_loss
        else:
            return total_loss, pred_mask, pred_ss, gt_label, pred_label


if __name__ == "__main__":
    import argparse

    from transformers import BertTokenizer, BertModel
    from data.SROIE_dataset import load_train_dataset
    from pipeline.train_val_utils import validate
    from pipeline.criteria import token_F1_criteria, BIO_F1_criteria

    TAG_TO_IDX = {
        "O": 0,
        "B-company": 1,
        "I-company": 2,
        "B-date": 3,
        "I-date": 4,
        "B-address": 5,
        "I-address": 6,
        "B-total": 7,
        "I-total": 8,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="dir to data root")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    bert_version = "private_bert-base-uncased"

    tokenizer = BertTokenizer.from_pretrained(bert_version)

    train_loader, val_loader = load_train_dataset(
        root=args.root,
        batch_size=2,
        num_workers=0,
        tokenizer=tokenizer,
    )

    model = ViBERTgridNet(
        num_classes=5,
        image_mean=[0.9247, 0.9238, 0.9229],
        image_std=[0.1530, 0.1542, 0.1533],
        image_min_size=[320, 416, 512, 608, 704],
        image_max_size=800,
        test_image_min_size=512,
        tokenizer=tokenizer,
        backbone="resnet_34_fpn",
        num_hard_positive_main_1=2,
        num_hard_negative_main_1=2,
        num_hard_positive_main_2=2,
        num_hard_negative_main_2=2,
        loss_aux_sample_list=[128, 256, 128],
        num_hard_positive_aux=2,
        num_hard_negative_aux=2,
        bert_model=bert_version,
        classifier_mode="full",
        tag_to_idx=TAG_TO_IDX,
        ohem_random=True,
    )
    model = model.to(device)

    # validate(
    #     model=model,
    #     validate_loader=val_loader,
    #     device=device,
    #     epoch=0,
    #     logger=None,
    #     distributed=False,
    #     iter_msg=True,
    #     eval_mode="seq_and_str",
    #     tag_to_idx=TAG_TO_IDX,
    # )

    train_batch = next(iter(train_loader))
    for train_batch in train_loader:
        (
            image_list,
            seg_indices,
            token_classes,
            ocr_coors,
            ocr_corpus,
            mask,
        ) = train_batch
        image_list = tuple(image.to(device) for image in image_list)
        seg_indices = tuple(class_label.to(device) for class_label in seg_indices)
        token_classes = tuple(
            pos_neg_label.to(device) for pos_neg_label in token_classes
        )
        ocr_coors = tuple(ocr_coor.to(device) for ocr_coor in ocr_coors)
        ocr_corpus = ocr_corpus.to(device)
        mask = mask.to(device)

        model.train()
        total_loss = model(
            image_list, seg_indices, token_classes, ocr_coors, ocr_corpus, mask
        )

        total_loss.backward()

        model.eval()
        total_loss, pred_mask, pred_ss, gt_label, pred_label = model(
            image_list, seg_indices, token_classes, ocr_coors, ocr_corpus, mask
        )

        p, r, f, report = BIO_F1_criteria({pred_label: gt_label}, TAG_TO_IDX)
        # eval_result = token_F1_criteria({pred_label: gt_label})
        print(report)

    print(
        "debug finished, total_loss = {} result: {}".format(
            total_loss.item(), [p, r, f]
        )
    )
