import os
import argparse
import yaml
import tqdm

import torch
from transformers import BertTokenizer, RobertaTokenizer

from model.ViBERTgrid_net import ViBERTgridNet
from data.SROIE_dataset import load_test_data
from pipeline.criteria import BIO_F1_criteria

from typing import Iterable, Dict


TAG_TO_IDX = {
    "O": 0,
    "B-question": 1,
    "B-answer": 2,
    "B-header": 3,
}


@torch.no_grad()
def evaluation_FUNSD(
    model: torch.nn.Module,
    evaluation_loader: Iterable,
    device: torch.device,
):

    model.eval()
    pred_gt_dict = dict()
    for evaluation_batch in tqdm.tqdm(evaluation_loader):
        (
            image_list,
            seg_indices,
            token_classes,
            ocr_coors,
            ocr_corpus,
            mask,
            ocr_text,
            key_dict,
        ) = evaluation_batch

        assert (
            len(key_dict) == 1
        ), f"batch size in evaluation must be 1, {len(key_dict)} given"

        image_list = tuple(image.to(device) for image in image_list)
        seg_indices = tuple(seg_index.to(device) for seg_index in seg_indices)
        token_classes = tuple(token_class.to(device) for token_class in token_classes)
        ocr_coors = tuple(ocr_coor.to(device) for ocr_coor in ocr_coors)
        ocr_corpus = ocr_corpus.to(device)
        mask = mask.to(device)

        pred_label: torch.Tensor
        _, _, _, gt_label, pred_label = model(
            image_list, seg_indices, token_classes, ocr_coors, ocr_corpus, mask
        )

        pred_gt_dict.update({pred_label.detach(): gt_label.detach()})

    p, r, f, report = BIO_F1_criteria(
        pred_gt_dict=pred_gt_dict, tag_to_idx=TAG_TO_IDX, average="macro"
    )

    return p, r, f, report


def main(args):
    with open(args.config, "r") as c:
        hyp = yaml.load(c, Loader=yaml.FullLoader)

    device = hyp["device"]
    num_workers = hyp["num_workers"]

    weights = hyp["weights"]

    data_root = hyp["data_root"]
    num_classes = hyp["num_classes"]
    image_mean = hyp["image_mean"]
    image_std = hyp["image_std"]
    image_min_size = hyp["image_min_size"]
    image_max_size = hyp["image_max_size"]
    test_image_min_size = hyp["test_image_min_size"]

    bert_version = hyp["bert_version"]
    backbone = hyp["backbone"]
    grid_mode = hyp["grid_mode"]
    early_fusion_downsampling_ratio = hyp["early_fusion_downsampling_ratio"]
    roi_shape = hyp["roi_shape"]
    p_fuse_downsampling_ratio = hyp["p_fuse_downsampling_ratio"]
    late_fusion_fuse_embedding_channel = hyp["late_fusion_fuse_embedding_channel"]
    loss_weights = hyp["loss_weights"]
    loss_control_lambda = hyp["loss_control_lambda"]
    layer_mode = hyp["layer_mode"]

    classifier_mode = hyp["classifier_mode"]

    device = torch.device(device)

    print(f"==> loading tokenizer {bert_version}")
    if "bert-" in bert_version:
        tokenizer = BertTokenizer.from_pretrained(bert_version)
    elif "roberta-" in bert_version:
        tokenizer = RobertaTokenizer.from_pretrained(bert_version)
    print(f"==> tokenizer {bert_version} loaded")

    print(f"==> loading datasets")
    test_loader = load_test_data(
        root=os.path.join(data_root, "test"),
        num_workers=num_workers,
        tokenizer=tokenizer,
    )
    print(f"==> dataset loaded")

    print(f"==> creating model {backbone} | {bert_version}")
    model = ViBERTgridNet(
        num_classes=num_classes,
        image_mean=image_mean,
        image_std=image_std,
        image_min_size=image_min_size,
        image_max_size=image_max_size,
        test_image_min_size=test_image_min_size,
        bert_model=bert_version,
        tokenizer=tokenizer,
        backbone=backbone,
        grid_mode=grid_mode,
        early_fusion_downsampling_ratio=early_fusion_downsampling_ratio,
        roi_shape=roi_shape,
        p_fuse_downsampling_ratio=p_fuse_downsampling_ratio,
        late_fusion_fuse_embedding_channel=late_fusion_fuse_embedding_channel,
        loss_weights=loss_weights,
        loss_control_lambda=loss_control_lambda,
        classifier_mode=classifier_mode,
        ohem_random=True,
        layer_mode=layer_mode,
        train=False,
    )
    model = model.to(device)
    print(f"==> model created")

    if weights != "":
        print("==> loading pretrained")
        checkpoint = torch.load(weights, map_location="cpu")["model"]
        model_weights = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(model_weights, strict=False)
        print(f"==> pretrained loaded")
    else:
        raise ValueError("weights must be provided")

    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("total number of parameters: " + str(k))

    print("==> testing...")
    p, r, f, report = evaluation_FUNSD(
        model=model,
        evaluation_loader=test_loader,
        device=device,
    )
    print(report)
    print(f"precision [{p}] | recall [{r}] | F1 [{f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="directory to config file",
    )

    args = parser.parse_args()

    main(args)
