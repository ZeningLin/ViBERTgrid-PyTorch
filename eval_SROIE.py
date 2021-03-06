import os
import re
import argparse
import yaml
import tqdm
import json

import torch
from transformers import BertTokenizer, RobertaTokenizer

from model.ViBERTgrid_net import ViBERTgridNet
from data.SROIE_dataset import load_test_data

from typing import Iterable, Dict


SROIE_CLASS_LIST = ["others", "company", "date", "address", "total"]


def SROIE_result_filter(raw_string: str, class_index: int):
    if class_index == 1:
        # company
        return raw_string
    elif class_index == 2:
        # date
        date_re = re.compile(
            r"((?i)(?:[12][0-9]|3[01]|0*[1-9])(?P<sep>[- \/.\\])(?P=sep)*(?:1[012]|0*[1-9]|jan(?:uary)?|feb("
            r"?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov("
            r"?:ember)?|dec(?:ember)?)(?P=sep)+(?:19|20)\d\d|(?:[12][0-9]|3[01]|0*[1-9])(?P<sep2>[- \/.\\])("
            r"?P=sep2)*(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul("
            r"?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P=sep2)+\d\d|(?:1[012]|0*["
            r"1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep("
            r"?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P<sep3>[- \/.\\])(?P=sep3)*(?:[12][0-9]|3[01]|0*["
            r"1-9])(?P=sep3)+(?:19|20)\d\d|(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr("
            r"?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)("
            r"?P<sep4>[- \/.\\])(?P=sep4)*(?:[12][0-9]|3[01]|0*[1-9])(?P=sep4)+\d\d|(?:19|20)\d\d(?P<sep5>[- \/.\\])("
            r"?P=sep5)*(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul("
            r"?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P=sep5)+(?:[12][0-9]|3["
            r"01]|0*[1-9])|\d\d(?P<sep6>[- \/.\\])(?P=sep6)*(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar("
            r"?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec("
            r"?:ember)?)(?P=sep6)+(?:[12][0-9]|3[01]|0*[1-9])|(?:[12][0-9]|3[01]|0*[1-9])(?:jan(?:uary)?|feb("
            r"?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov("
            r"?:ember)?|dec(?:ember)?)(?:19|20)\d\d|(?:[12][0-9]|3[01]|0*[1-9])(?:jan(?:uary)?|feb(?:ruary)?|mar("
            r"?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec("
            r"?:ember)?)\d\d|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug("
            r"?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[12][0-9]|3[01]|0*[1-9])("
            r"?:19|20)\d\d|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug("
            r"?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[12][0-9]|3[01]|0*[1-9])\d\d|("
            r"?:19|20)\d\d(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug("
            r"?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[12][0-9]|3[01]|0*[1-9])|\d\d(?:jan("
            r"?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct("
            r"?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[12][0-9]|3[01]|0*[1-9])|(?:[12][0-9]|3[01]|0[1-9])(?:1[012]|0["
            r"1-9])(?:19|20)\d\d|(?:1[012]|0[1-9])(?:[12][0-9]|3[01]|0[1-9])(?:19|20)\d\d|(?:19|20)\d\d(?:1[012]|0["
            r"1-9])(?:[12][0-9]|3[01]|0[1-9])|(?:1[012]|0[1-9])(?:[12][0-9]|3[01]|0[1-9])\d\d|(?:[12][0-9]|3[01]|0["
            r"1-9])(?:1[012]|0[1-9])\d\d|\d\d(?:1[012]|0[1-9])(?:[12][0-9]|3[01]|0[1-9]))"
        )
        date_match = date_re.match(raw_string)
        if date_match is not None:
            return date_match[0]
        else:
            return None
    elif class_index == 3:
        # address
        return raw_string
    elif class_index == 4:
        # total
        total_re = re.compile("^\d+(\.\d+)?$")
        total_match = total_re.search(raw_string)
        if total_match is not None:
            return total_match[0]
        else:
            return None


@torch.no_grad()
def evaluation_SROIE(
    model: torch.nn.Module,
    evaluation_loader: Iterable,
    device: torch.device,
    tresh: float = 0,
):
    num_classes = len(SROIE_CLASS_LIST)

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
                if curr_class_str.endswith("-"):
                    curr_class_str += ocr_text[0][seg_index]
                else:
                    curr_class_str += " " + ocr_text[0][seg_index]
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

        # pred_label = pred_label.softmax(dim=1).argmax(dim=1).int()
        # pred_key_list = ["" for _ in range(num_classes)]
        # for seg_index in range(pred_label.shape[0]):
        #     pred_class = pred_label[seg_index].item()
        #     if pred_key_list[pred_class].endswith("-"):
        #         pred_key_list[pred_class] += ocr_text[0][seg_index]
        #     elif pred_key_list[pred_class] == "":
        #         pred_key_list[pred_class] += ocr_text[0][seg_index]
        #     else:
        #         pred_key_list[pred_class] += " " + ocr_text[0][seg_index]

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
        for class_index in range(num_classes):
            if class_index == 0:
                continue
            curr_pred_str = pred_key_list[class_index]
            curr_pred_str = SROIE_result_filter(curr_pred_str, class_index)
            curr_class_name = SROIE_CLASS_LIST[class_index]
            curr_gt_str = key_dict[0][curr_class_name]
            if len(curr_pred_str) != 0:
                curr_num_det += 1
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

        precision = (
            float(0) if (curr_num_det) == 0 else float(precision_accum) / (curr_num_det)
        )
        recall = (
            float(1)
            if (num_classes - 1) == 0
            else float(recall_accum) / (num_classes - 1)
        )
        hmean = (
            0
            if (precision + recall) == 0
            else 2.0 * precision * recall / (precision + recall)
        )

        method_recall_sum += recall_accum
        method_precision_sum += precision_accum
        num_gt += num_classes - 1
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
        work_mode="eval",
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
    res_dict = evaluation_SROIE(
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
