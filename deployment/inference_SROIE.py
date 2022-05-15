import time
import re
import torch

from transformers import BertTokenizer

from deployment.inference_preporcessing import generate_batch
from model.ViBERTgrid_net import ViBERTgridNet

from typing import Tuple


SROIE_CLASS_LIST = ["others", "company", "date", "address", "total"]


def SROIE_date_filter(raw_result: str):
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
    date_match = date_re.match(raw_result)
    if date_match is not None:
        return date_match[0]
    else:
        return ""


def SROIE_total_filter(raw_result: str):
    total_re = re.compile("^\d+(\.\d+)?$")
    total_match = total_re.search(raw_result)
    if total_match is not None:
        return total_match[0]
    else:
        return ""


@torch.no_grad()
def SROIE_postprocessing(pred_label: torch.Tensor, num_classes: int, ocr_text: Tuple):
    pred_all_list = [list() for _ in range(num_classes)]
    curr_class_str = ""
    curr_class_score = 0.0
    curr_class_seg_len = 0
    prev_class = -1
    for seg_index in range(pred_label.shape[0]):
        curr_pred_logits = pred_label[seg_index].softmax(dim=0)
        curr_pred_class: torch.Tensor = curr_pred_logits.argmax(dim=0)
        curr_pred_score = curr_pred_logits[curr_pred_class].item()

        if curr_pred_class == prev_class:
            if curr_class_str.endswith("-"):
                curr_class_str += ocr_text[0][seg_index]
            else:
                curr_class_str += ocr_text[0][seg_index] + " "

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

    pred_key_dict = {k: "" for k in SROIE_CLASS_LIST[1:]}
    for class_index, class_all_result in enumerate(pred_all_list):
        if class_index == 0:
            continue
        curr_class_str = SROIE_CLASS_LIST[class_index]
        if class_all_result is None or len(class_all_result) == 0:
            continue

        max_score = 0
        max_index = 0
        for curr_index, candidates in enumerate(class_all_result):
            curr_score = candidates[1]
            if curr_score > max_score:
                max_score = curr_score
                max_index = curr_index

        curr_result = class_all_result[max_index][0]
        if class_index == 2:
            curr_result = SROIE_date_filter(curr_result)
        elif class_index == 4:
            curr_result = SROIE_total_filter(curr_result)
        pred_key_dict[curr_class_str] = curr_result

    return pred_key_dict


@torch.no_grad()
def model_inference(
    model: ViBERTgridNet,
    batch: tuple,
    num_classes: int,
):
    model.eval()
    (
        image_list,
        seg_indices,
        ocr_coors,
        ocr_corpus,
        mask,
        ocr_text,
    ) = batch

    start_time = time.time()
    pred_label: torch.Tensor = model.inference(
        image=image_list,
        seg_indices=seg_indices,
        coors=ocr_coors,
        corpus=ocr_corpus,
        mask=mask,
    )
    print(f"Model Inference Time {time.time() - start_time}s")

    pred_result = SROIE_postprocessing(
        pred_label=pred_label, num_classes=num_classes, ocr_text=ocr_text
    )

    return pred_result


def inference_pipe(
    model: torch.nn.Module,
    ocr_url: str,
    tokenizer: BertTokenizer,
    device: torch.DeviceObjType,
    num_classes: int,
    image_bytes: bytes = None,
    parse_mode: str = None,
):
    batch = generate_batch(
        image_bytes=image_bytes,
        ocr_url=ocr_url,
        tokenizer=tokenizer,
        device=device,
        parse_mode=parse_mode,
    )

    return model_inference(
        model=model,
        batch=batch,
        num_classes=num_classes,
    )


if __name__ == "__main__":
    import json
    import argparse
    from deployment.module_load import inference_init

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="directory to config file",
    )
    parser.add_argument("--img", type=str, required=True, help="dir to images")
    args = parser.parse_args()

    image_dir = args.img
    with open(image_dir, "rb") as image_file:
        image_bytes = image_file.read()

    (
        MODEL,
        OCR_URL,
        TOKENIZER,
        DEVICE,
        NUM_CLASSES,
        PARSE_MODE,
    ) = inference_init(dir_config=args.config)

    result = inference_pipe(
        MODEL,
        OCR_URL,
        TOKENIZER,
        DEVICE,
        NUM_CLASSES,
        image_bytes=image_bytes,
        parse_mode=PARSE_MODE,
    )

    with open(image_dir.replace(".jpg", ".json"), "w") as f:
        json.dump(result, f, ensure_ascii=False)
