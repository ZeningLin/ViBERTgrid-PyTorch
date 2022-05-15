import time
import yaml
import argparse

import torch
from transformers import BertTokenizer
from ltp import LTP

from model.ViBERTgrid_net import ViBERTgridNet


def inference_init(
    dir_config: str = "./deployment/config/network_config.yaml"
):
    with open(dir_config, "r") as c:
        hyp = yaml.load(c, Loader=yaml.FullLoader)

    ocr_url = hyp["ocr_url"]
    parse_mode = hyp["parse_mode"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    weights = hyp["weights"]
    num_classes = hyp["num_classes"]
    image_mean = hyp["image_mean"]
    image_std = hyp["image_std"]

    bert_version = hyp["bert_version"]
    backbone = hyp["backbone"]
    grid_mode = hyp["grid_mode"]
    early_fusion_downsampling_ratio = hyp["early_fusion_downsampling_ratio"]
    roi_shape = hyp["roi_shape"]
    p_fuse_downsampling_ratio = hyp["p_fuse_downsampling_ratio"]
    late_fusion_fuse_embedding_channel = hyp["late_fusion_fuse_embedding_channel"]

    layer_mode = hyp["layer_mode"]
    classifier_mode = hyp["classifier_mode"]

    print("[LOADING] bert tokenizer")
    start_time = time.time()
    tokenizer = BertTokenizer.from_pretrained(bert_version)
    print(f"[LOADED] bert tokenizer, time used {time.time() - start_time}s")

    print(f"[LOADING] model {backbone} | {bert_version}")
    start_time = time.time()
    model = ViBERTgridNet(
        num_classes=num_classes,
        image_mean=image_mean,
        image_std=image_std,
        image_min_size=[512],
        image_max_size=800,
        test_image_min_size=512,
        bert_model=bert_version,
        tokenizer=tokenizer,
        backbone=backbone,
        grid_mode=grid_mode,
        early_fusion_downsampling_ratio=early_fusion_downsampling_ratio,
        roi_shape=roi_shape,
        p_fuse_downsampling_ratio=p_fuse_downsampling_ratio,
        late_fusion_fuse_embedding_channel=late_fusion_fuse_embedding_channel,
        classifier_mode=classifier_mode,
        tag_to_idx=None,
        layer_mode=layer_mode,
        work_mode="inference",
    )

    model_param_list = list(model.parameters())
    k = 0
    for i in model_param_list:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("total number of parameters: " + str(k))

    if weights != "":
        print("[LOADING] pretrained weights")
        checkpoint = torch.load(weights, map_location="cpu")["model"]
        model_weights = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(model_weights, strict=False)
        print(f"[LOADED] pretrained weights")
    else:
        raise ValueError("weights must be provided")

    model = model.to(device)
    print(f"[LOADED] model {backbone} | {bert_version}")
    print(f"time used: {time.time() - start_time}s")

    return model, ocr_url, tokenizer, device, num_classes, parse_mode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="dir to config file")
    args = parser.parse_args()

    model, ocr_url, tokenizer, device, num_classes, parse_mode = inference_init(args)
