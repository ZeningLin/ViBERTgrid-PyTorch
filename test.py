import argparse
import yaml

import torch
from transformers import BertTokenizer, RobertaTokenizer

from model.ViBERTgrid_net import ViBERTgridNet
from data.SROIE_dataset import load_test_data
from pipeline.train_val_utils import validate


def inference(args):
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
    roi_align_output_reshape = hyp["roi_align_output_reshape"]
    late_fusion_fuse_embedding_channel = hyp["late_fusion_fuse_embedding_channel"]
    loss_weights = hyp["loss_weights"]
    loss_control_lambda = hyp["loss_control_lambda"]

    device = torch.device(device)

    print(f"==> loading tokenizer {bert_version}")
    if "bert-" in bert_version:
        tokenizer = BertTokenizer.from_pretrained(bert_version)
    elif "roberta-" in bert_version:
        tokenizer = RobertaTokenizer.from_pretrained(bert_version)
    print(f"==> tokenizer {bert_version} loaded")

    print(f"==> loading datasets")
    test_loader = load_test_data(
        root=data_root,
        num_workers=num_workers,
        tokenizer=tokenizer,
    )
    batch = next(iter(test_loader))
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
        tokenizer=None,
        backbone=backbone,
        grid_mode=grid_mode,
        early_fusion_downsampling_ratio=early_fusion_downsampling_ratio,
        roi_shape=roi_shape,
        p_fuse_downsampling_ratio=p_fuse_downsampling_ratio,
        roi_align_output_reshape=roi_align_output_reshape,
        late_fusion_fuse_embedding_channel=late_fusion_fuse_embedding_channel,
        loss_weights=loss_weights,
        loss_control_lambda=loss_control_lambda,
    )
    model = model.to(device)
    print(f"==> model created")

    if weights != "":
        print("==> loading pretrained")
        checkpoint = torch.load(weights, map_location="cpu")["model"]
        model_weights = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(model_weights)
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
    validate(
        model=model,
        validate_loader=test_loader,
        device=device,
        epoch=0,
        logger=None,
        distributed=False,
        iter_msg=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="directory to config file",
    )

    args = parser.parse_args()

    inference(args)
