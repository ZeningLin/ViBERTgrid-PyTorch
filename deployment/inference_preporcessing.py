import io
import requests
import PIL.Image as Image

import torch
import torchvision
from transformers import BertTokenizer
from ltp import LTP

from typing import Dict, List


def ocr_parsing_eng_line(api_return_result: Dict) -> Dict:
    status_code = api_return_result["code"]
    return_text_list = list()
    return_coor_list = list()
    if status_code == 200:
        line_info_list = api_return_result["result"]["lines"]
        for line_info in line_info_list:
            text_str: str = line_info["text"]
            line_coor = line_info["position"]
            return_text_list.append(text_str)
            return_coor_list.append(
                [line_coor[0], line_coor[1], line_coor[2], line_coor[5]]
            )

    return status_code, return_text_list, return_coor_list


def ocr_parsing_eng_word(api_return_result: Dict) -> Dict:
    status_code = api_return_result["code"]
    return_text_list = list()
    return_coor_list = list()
    if status_code == 200:
        line_info_list = api_return_result["result"]["lines"]

        for line_info in line_info_list:
            text_str: str = line_info["text"]
            char_coor_list: List[List[int]] = line_info["char_positions"]
            text_word_list = text_str.split()
            char_start_index = 0
            for word in text_word_list:
                word_len = len(word)
                char_end_index = char_start_index + word_len
                first_coor = char_coor_list[char_start_index]
                end_coor = char_coor_list[char_end_index]
                return_text_list.append(word)
                return_coor_list.append(
                    [first_coor[0], first_coor[1], end_coor[2], end_coor[5]]
                )
                char_start_index = char_end_index + 1

    return status_code, return_text_list, return_coor_list


def ocr_parsing_chn_char(api_return_result: Dict) -> Dict:
    status_code = api_return_result["code"]
    return_text_list = list()
    return_coor_list = list()
    if status_code == 200:
        line_info_list = api_return_result["result"]["lines"]

        for line_info in line_info_list:
            text_str = line_info["text"]
            char_coor_list = line_info["char_positions"]
            for text_char, char_coor in zip(text_str, char_coor_list):
                return_text_list.append(text_char)
                return_coor_list.append(
                    [char_coor[0], char_coor[1], char_coor[4], char_coor[5]]
                )

    return status_code, return_text_list, return_coor_list


def ocr_parsing_chn_ltp(api_return_result: Dict):
    status_code = api_return_result["code"]
    return_text_list = list()
    return_coor_list = list()
    ltp = LTP()
    if status_code == 200:
        line_info_list = api_return_result["result"]["lines"]

        for line_info in line_info_list:
            text_str = line_info["text"]
            char_coor_list = line_info["char_positions"]

            splitted_text_str = ltp.seg([text_str])[0][0]
            start_index = 0
            for seg in splitted_text_str:
                curr_len = len(seg)
                end_index = start_index + curr_len
                curr_str = seg
                curr_coor = char_coor_list[start_index:end_index]

                left_set = set()
                top_set = set()
                right_set = set()
                bot_set = set()
                for coor in curr_coor:
                    left_set.add(coor[0])
                    top_set.add(coor[1])
                    right_set.add(coor[2])
                    bot_set.add(coor[3])

                curr_left = min(left_set)
                curr_right = max(right_set)
                curr_top = min(top_set)
                curr_bot = max(bot_set)

                return_text_list.append(curr_str)
                return_coor_list.append([curr_left, curr_right, curr_top, curr_bot])

    return status_code, return_text_list, return_coor_list


def ocr_extraction(image_bytes, ocr_url: str, parse_mode: str) -> Dict:
    PARSE_MODE_DICT = {
        "eng_line": ocr_parsing_eng_line,
        "eng_word": ocr_parsing_eng_word,
        "chn_char": ocr_parsing_chn_char,
        "chn_ltp": ocr_parsing_chn_ltp,
    }

    headers = {"Content-Type": "application/octet-stream", "accept": "application/json"}

    api_return_result = dict()
    api_return_result["code"] = -1

    try:
        res = requests.post(url=ocr_url, data=image_bytes, headers=headers)
        if res.status_code == 200:
            api_return_result = res.json()
    except Exception as e:
        print(f"[ERROR] ocr engine failed, {e}")

    return PARSE_MODE_DICT[parse_mode](api_return_result)


def generate_batch(
    image_bytes: bytes,
    ocr_url: str,
    tokenizer: BertTokenizer,
    device: torch.device,
    parse_mode: str = None,
):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')

    status_code, return_text_list, return_coor_list = ocr_extraction(
        image_bytes=image_bytes, ocr_url=ocr_url, parse_mode=parse_mode
    )

    if status_code != 200:
        return

    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image = transforms(image)

    ocr_tokens = []
    seg_indices = []
    for seg_index, text in enumerate(return_text_list):
        if text == "":
            continue
        curr_tokens = tokenizer.tokenize(text)
        for i in range(len(curr_tokens)):
            ocr_tokens.append(curr_tokens[i])
            seg_indices.append(seg_index)

    ocr_corpus = tokenizer.convert_tokens_to_ids(ocr_tokens)
    ocr_corpus = torch.tensor(ocr_corpus, dtype=torch.long, device=device)
    mask = torch.ones(ocr_corpus.shape, dtype=torch.int, device=device)

    return (
        (image.to(device),),
        (torch.tensor(seg_indices, dtype=torch.int),),
        (torch.tensor(return_coor_list, dtype=torch.long, device=device),),
        ocr_corpus.unsqueeze(0),
        mask.unsqueeze(0),
        (return_text_list,),
    )
