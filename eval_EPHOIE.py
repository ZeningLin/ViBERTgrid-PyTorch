import os
import re
import argparse
import yaml
import tqdm
import json

import torch
from transformers import BertTokenizer, RobertaTokenizer

from model.ViBERTgrid_net import ViBERTgridNet
from data.EPHOIE_dataset import load_test_data

from typing import Iterable, Dict


EPHOIE_CLASS_LIST = [
    "其他",
    "年级",
    "科目",
    "学校",
    "考试时间",
    "班级",
    "姓名",
    "考号",
    "分数",
    "座号",
    "学号",
    "准考证号",
]

FILTER_WORD_LIST = [
    "年级",
    "科目",
    "学校",
    "考试时间",
    "班级",
    "姓名",
    "考号",
    "分数",
    "座号",
    "学号",
    "准考证号",
    "：",
    ":",
    "得分",
    "等级",
    "班次",
]

SUBJECT_LIST = [
    "语文",
    "数学",
    "英语",
    "政治",
    "道德与法治",
    "思想品德",
    "历史",
    "地理",
    "生物",
    "化学",
    "物理",
    "文综",
    "文科综合",
    "理综",
    "理科综合",
    "科学",
    "历史与社会",
    "品德与社会",
    "语文",
    "历史与社会·道德与法治",
    "数据的分析",
    "地理生物",
]


def normal_filter(raw_string: str):
    filter_index_list = list()
    for filter_word in FILTER_WORD_LIST:
        curr_len = len(filter_word)
        match_index = raw_string.find(filter_word)
        if match_index < 0:
            continue
        for i in range(curr_len):
            filter_index_list.append(match_index + i)

    return filter_index_list


def subject_category_filter(raw_string: str):
    for item in SUBJECT_LIST:
        if raw_string.find(item) > 0:
            return item

    return ""


def grade_category_filter(raw_string: str):
    filter_index_list = list()
    find_school_key = raw_string.find("年级")
    if find_school_key >= 0:
        if find_school_key == 0:
            # 一般以年级开头的基本都是key，直接干掉
            filter_index_list.append(0)
            filter_index_list.append(1)

    for filter_word in FILTER_WORD_LIST:
        curr_len = len(filter_word)
        match_index = raw_string.find(filter_word)
        if match_index < 0:
            continue
        for i in range(curr_len):
            filter_index_list.append(match_index + i)

    return filter_index_list


def school_category_filter(raw_string: str):
    filter_index_list = list()
    find_school_key = raw_string.find("学校")
    if find_school_key >= 0:
        if find_school_key == 0:
            # 一般以学校开头的基本都是key，直接干掉
            filter_index_list.append(0)
            filter_index_list.append(1)

    for filter_word in FILTER_WORD_LIST:
        curr_len = len(filter_word)
        match_index = raw_string.find(filter_word)
        if match_index < 0:
            continue
        for i in range(curr_len):
            filter_index_list.append(match_index + i)

    return filter_index_list


def EPHOIE_result_filter(raw_string: str, class_index: int):
    if class_index == 1:
        filter_index_list = grade_category_filter(raw_string)
    elif class_index == 2:
        filter_index_list = subject_category_filter(raw_string)
    elif class_index == 3:
        filter_index_list = school_category_filter(raw_string)
    else:
        filter_index_list = normal_filter(raw_string)

    filtered_str = ""
    for char_index, (char) in enumerate(raw_string):
        if char_index in filter_index_list:
            continue

        filtered_str += char

    return filtered_str


@torch.no_grad()
def evaluation_EPHOIE(
    model: torch.nn.Module,
    evaluation_loader: Iterable,
    device: torch.device,
    tresh: float = 0,
):
    num_classes = len(EPHOIE_CLASS_LIST)

    num_gt = 0.0
    num_det = 0.0
    method_recall_sum = 0
    method_precision_sum = 0
    per_sample_metrics = dict()

    model.eval()
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
        _, _, _, _, pred_label = model(
            image_list, seg_indices, token_classes, ocr_coors, ocr_corpus, mask
        )

        pred_all_list = [list() for _ in range(num_classes)]
        curr_class_str = ""
        curr_class_score = 0.0
        curr_class_seg_len = 0
        prev_class = -1
        for seg_index in range(pred_label.shape[0]):
            curr_pred_logits = pred_label[seg_index].softmax(dim=0)
            curr_pred_class: torch.Tensor = curr_pred_logits.argmax(dim=0)
            curr_pred_score = curr_pred_logits[curr_pred_class].item()
            if curr_pred_score < tresh:
                curr_pred_class = 0

            if curr_pred_class == prev_class:
                curr_class_str += ocr_text[0][seg_index]
                curr_class_score += curr_pred_score
                curr_class_seg_len += 1
            else:
                if prev_class >= 0:
                    pred_all_list[prev_class].append(
                        (curr_class_str, (curr_class_score / curr_class_seg_len))
                    )

                curr_class_str = ocr_text[0][seg_index]
                curr_class_score = curr_pred_score
                curr_class_seg_len = 1

            if seg_index == pred_label.shape[0] - 1:
                pred_all_list[prev_class].append(
                    (curr_class_str, (curr_class_score / curr_class_seg_len))
                )

            prev_class = curr_pred_class

        pred_key_list = list()
        for class_all_result in pred_all_list:
            if class_all_result is None or len(class_all_result) == 0:
                pred_key_list.append("")
                continue

            max_score = 0
            max_index = 0
            for curr_index, candidates in enumerate(class_all_result):
                curr_score = candidates[1]
                if curr_score > max_score:
                    max_score = curr_score
                    max_index = curr_index

            pred_key_list.append(class_all_result[max_index][0])

        recall = 0
        precision = 0
        recall_accum = 0.0
        precision_accum = 0.0
        filename = key_dict[0]["filename"]
        log = dict()
        curr_log = " ".join(
            ["pred_key: [{pred_key}]", "gt_key: [{gt_key}]", "status: {status}"]
        )
        curr_num_det = 0.0
        curr_num_gt = 0.0
        for class_index in range(num_classes):
            if class_index == 0:
                continue
            curr_pred_str = pred_key_list[class_index]
            curr_pred_str = EPHOIE_result_filter(curr_pred_str, class_index)
            curr_class_name = EPHOIE_CLASS_LIST[class_index]
            curr_gt_str = key_dict[0][curr_class_name]
            if len(curr_pred_str) != 0:
                curr_num_det += 1
            if len(curr_gt_str) != 0:
                curr_num_gt += 1
                if curr_pred_str == curr_gt_str:
                    recall_accum += 1
                    precision_accum += 1
                    log[curr_class_name] = curr_log.format(
                        pred_key=curr_pred_str, gt_key=curr_gt_str, status="CORRECT"
                    )
                else:
                    log[curr_class_name] = curr_log.format(
                        pred_key=curr_pred_str, gt_key=curr_gt_str, status="ERROR"
                    )
            else:
                if len(curr_pred_str) != 0:
                    log[curr_class_name] = curr_log.format(
                        pred_key=curr_pred_str, gt_key=curr_gt_str, status="ERROR"
                    )

        precision = (
            float(0) if (curr_num_det) == 0 else float(precision_accum) / (curr_num_det)
        )
        recall = (
            float(1) if (num_classes - 1) == 0 else float(recall_accum) / (curr_num_gt)
        )
        hmean = (
            0
            if (precision + recall) == 0
            else 2.0 * precision * recall / (precision + recall)
        )

        method_recall_sum += recall_accum
        method_precision_sum += precision_accum
        num_gt += curr_num_gt
        num_det += curr_num_det

        per_sample_metrics[filename] = {
            "precision": precision,
            "recall": recall,
            "hmean": hmean,
            "correct": recall_accum,
            "log": log,
        }

    method_recall = 0 if num_gt == 0 else method_recall_sum / num_gt
    method_precision = 0 if num_det == 0 else method_precision_sum / num_det
    method_Hmean = (
        0
        if method_recall + method_precision == 0
        else 2 * method_recall * method_precision / (method_recall + method_precision)
    )
    method_metrics = {
        "precision": method_precision,
        "recall": method_recall,
        "hmean": method_Hmean,
    }

    res_dict = {
        "method": method_metrics,
        "per_sample": per_sample_metrics,
    }

    return res_dict


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
        root=os.path.join(data_root),
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
    res_dict = evaluation_EPHOIE(
        model=model,
        evaluation_loader=test_loader,
        device=device,
    )

    precision = res_dict["method"]["precision"]
    recall = res_dict["method"]["recall"]
    hmean = res_dict["method"]["hmean"]

    print(f"precision[{precision:.4f}] recall[{recall:.4f}] F1[{hmean:.4f}]")

    if not os.path.exists("result"):
        os.mkdir("result")
    dir_save = os.path.basename(weights)
    dir_save = os.path.join("result", dir_save.replace(".pth", ".json"))
    with open(dir_save, "w") as f:
        json.dump(res_dict, f, ensure_ascii=False)


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
