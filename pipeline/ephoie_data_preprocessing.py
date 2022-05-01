import os
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shutil import copy
from tqdm import tqdm

from typing import Any, Dict, List, Tuple

"""
标签规则
{
    '其他': 0,
    '年级': 1, 
    '科目': 2, 
    '学校': 3, 
    '考试时间': 4, 
    '班级': 5, 
    '姓名': 6, 
    '考号': 7, 
    '分数': 8, 
    '座号': 9, 
    '学号': 10, 
    '准考证号': 11
}

"""


TAG_TO_IDX = {
    "O": 0,
    "B-grade": 1,
    "I-grade": 2,
    "B-subject": 3,
    "I-subject": 4,
    "B-school": 5,
    "I-school": 6,
    "B-testtime": 7,
    "I-testtime": 8,
    "B-class": 9,
    "I-class": 10,
    "B-name": 11,
    "I-name": 12,
    "B-testno": 13,
    "I-testno": 14,
    "B-score": 15,
    "I-score": 16,
    "B-seatno": 17,
    "I-seatno": 18,
    "B-studentno": 19,
    "I-studentno": 20,
    "B-testadmissionno": 21,
    "I-testadmissionno": 22,
}

IDX_TO_TAG = {v: k for k, v in TAG_TO_IDX.items()}


def generate_json(root_dir_txt_label: str, root_dir_json_label: str) -> None:
    """convert txt label to json format

    Parameters
    ----------
    txt_label_root : str
        root dir of txt labels
    json_label_root : str
        target root dir of json labels
    """
    assert os.path.exists(
        root_dir_txt_label
    ), f"The given txt_label_root {root_dir_txt_label} does not exists"
    if os.path.exists(root_dir_json_label):
        if os.listdir(root_dir_json_label) is not None:
            warnings.warn(
                "json files might already exist, make sure that you want to convert again"
            )
    if not os.path.exists(root_dir_json_label):
        os.mkdir(root_dir_json_label)

    txt_list = os.listdir(root_dir_txt_label)
    for file in tqdm(txt_list):
        copy(
            os.path.abspath(os.path.join(root_dir_txt_label, file)),
            os.path.abspath(
                os.path.join(root_dir_json_label, file.replace("txt", "json"))
            ),
        )


# def single_label_parser_full_seg(
#     dir_img: str,
#     dir_json_label: str,
#     dir_csv_label: str,
#     target_shape: Tuple[int] = None,
# ):
#     image = plt.imread(dir_img)
#     image_shape = image.shape

#     if target_shape is not None:
#         assert (
#             len(target_shape) == 2
#         ), f"target_shape can only contain 2 elements, {len(target_shape)} given"
#         pos_neg_label = np.zeros(target_shape, dtype=int)
#         class_label = np.zeros(target_shape, dtype=int)
#     else:
#         pos_neg_label = np.zeros((image_shape[0], image_shape[1]), dtype=int)
#         class_label = np.zeros((image_shape[0], image_shape[1]), dtype=int)

#     csv_label = pd.DataFrame(
#         columns=["left", "top", "right", "bot", "text", "data_class", "pos_neg"]
#     )

#     with open(dir_json_label, "rb") as json_f:
#         json_label: Dict = json.load(json_f)

#         for segment in json_label.values():
#             hor_candidate = segment["box"][::2]
#             ver_candidate = segment["box"][1::2]

#             left_coor = int(min(hor_candidate))
#             top_coor = int(min(ver_candidate))
#             right_coor = int(max(hor_candidate))
#             bot_coor = int(max(ver_candidate))
#             width = right_coor - left_coor

#             if target_shape is not None:
#                 scale_x = target_shape[0] / image_shape[0]
#                 scale_y = target_shape[1] / image_shape[1]
#                 left_coor *= scale_x
#                 right_coor *= scale_x
#                 top_coor *= scale_y
#                 bot_coor *= scale_y

#             seg_class = list()
#             char_pos_neg = 0

#             curr_row_dict = {
#                 "left": [left_coor],
#                 "top": [top_coor],
#                 "right": [right_coor],
#                 "bot": [bot_coor],
#                 "text": [],
#                 "data_class": [seg_class],
#                 "pos_neg": [char_pos_neg],
#             }


def single_label_parser_ltp(
    dir_img: str,
    dir_json_label: str,
    dir_csv_label: str,
    target_shape: Tuple[int] = None,
    discard_key: bool = False,
):
    from ltp import LTP

    image = plt.imread(dir_img)
    image_shape = image.shape

    if target_shape is not None:
        assert (
            len(target_shape) == 2
        ), f"target_shape can only contain 2 elements, {len(target_shape)} given"

    csv_label = pd.DataFrame(
        columns=["left", "top", "right", "bot", "text", "data_class", "pos_neg"]
    )

    ltp_splitter = LTP()

    with open(dir_json_label, "rb") as json_f:
        json_label: Dict = json.load(json_f)

        for segment in json_label.values():
            splitted, _ = ltp_splitter.seg([segment["string"]])
            num_char = len(segment["string"])
            num_seg = len(splitted[0])
            hor_candidate = segment["box"][::2]
            ver_candidate = segment["box"][1::2]

            left_coor = int(min(hor_candidate))
            top_coor = int(min(ver_candidate))
            right_coor = int(max(hor_candidate))
            bot_coor = int(max(ver_candidate))
            width = right_coor - left_coor

            if target_shape is not None:
                scale_x = target_shape[0] / image_shape[0]
                scale_y = target_shape[1] / image_shape[1]
                left_coor *= scale_x
                right_coor *= scale_x
                top_coor *= scale_y
                bot_coor *= scale_y

            char_width = (width + num_char - 1) // num_char
            curr_left = left_coor
            char_index = 0
            for seg_index in range(num_seg):
                num_curr_char = len(splitted[0][seg_index])
                curr_right = curr_left + char_width * num_curr_char

                seg_kv = segment["class"]
                if discard_key and seg_kv == "KEY":
                    seg_class = 0
                else:
                    seg_class = segment["tag"][char_index]

                char_pos_neg = 2 if (seg_class == 0) else 1

                curr_row_dict = {
                    "left": [curr_left],
                    "top": [top_coor],
                    "right": [curr_right],
                    "bot": [bot_coor],
                    "text": [splitted[0][seg_index]],
                    "data_class": [seg_class],
                    "pos_neg": [char_pos_neg],
                }
                curr_row_dataframe = pd.DataFrame(curr_row_dict)
                csv_label = pd.concat(
                    [csv_label, curr_row_dataframe], axis=0, ignore_index=True
                )

                curr_left = curr_right
                char_index += num_curr_char

    csv_label.to_csv(dir_csv_label)


def single_label_parser_char_BIO(
    dir_img: str,
    dir_json_label: str,
    dir_csv_label: str,
    target_shape: Tuple[int] = None,
    discard_key: bool = False,
):
    image = plt.imread(dir_img)
    image_shape = image.shape

    if target_shape is not None:
        assert (
            len(target_shape) == 2
        ), f"target_shape can only contain 2 elements, {len(target_shape)} given"

    csv_label = pd.DataFrame(
        columns=["left", "top", "right", "bot", "text", "data_class", "pos_neg"]
    )

    with open(dir_json_label, "rb") as json_f:
        json_label: Dict = json.load(json_f)

        prev_class = -1
        for segment in json_label.values():
            num_char = len(segment["string"])

            hor_candidate = segment["box"][::2]
            ver_candidate = segment["box"][1::2]

            left_coor = int(min(hor_candidate))
            top_coor = int(min(ver_candidate))
            right_coor = int(max(hor_candidate))
            bot_coor = int(max(ver_candidate))
            width = right_coor - left_coor

            if target_shape is not None:
                scale_x = target_shape[0] / image_shape[0]
                scale_y = target_shape[1] / image_shape[1]
                left_coor *= scale_x
                right_coor *= scale_x
                top_coor *= scale_y
                bot_coor *= scale_y

            char_width = (width + num_char - 1) // num_char
            curr_left = left_coor

            for char_index in range(num_char):
                curr_right = curr_left + char_width

                char_kv = segment["class"]
                if discard_key and char_kv == "KEY":
                    char_class = 0
                else:
                    char_class = segment["tag"][char_index]

                if char_class != 0:
                    if char_class != prev_class:
                        cvt_char_class = char_class * 2 - 1
                    else:
                        cvt_char_class = char_class * 2
                else:
                    cvt_char_class = 0

                prev_class = char_class

                char_pos_neg = 2 if (char_class == 0) else 1

                curr_row_dict = {
                    "left": [curr_left],
                    "top": [top_coor],
                    "right": [curr_right],
                    "bot": [bot_coor],
                    "text": [str(segment["string"][char_index])],
                    "data_class": [cvt_char_class],
                    "pos_neg": [char_pos_neg],
                    "class_str": [str(IDX_TO_TAG[cvt_char_class])],
                }
                curr_row_dataframe = pd.DataFrame(curr_row_dict)
                csv_label = pd.concat(
                    [csv_label, curr_row_dataframe], axis=0, ignore_index=True
                )

                curr_left = curr_right

    csv_label.to_csv(dir_csv_label)


def single_label_parser_char(
    dir_img: str,
    dir_json_label: str,
    dir_csv_label: str,
    target_shape: Tuple[int] = None,
    discard_key: bool = False,
):
    image = plt.imread(dir_img)
    image_shape = image.shape

    if target_shape is not None:
        assert (
            len(target_shape) == 2
        ), f"target_shape can only contain 2 elements, {len(target_shape)} given"

    csv_label = pd.DataFrame(
        columns=["left", "top", "right", "bot", "text", "data_class", "pos_neg"]
    )

    with open(dir_json_label, "rb") as json_f:
        json_label: Dict = json.load(json_f)

        for segment in json_label.values():
            num_char = len(segment["string"])

            hor_candidate = segment["box"][::2]
            ver_candidate = segment["box"][1::2]

            left_coor = int(min(hor_candidate))
            top_coor = int(min(ver_candidate))
            right_coor = int(max(hor_candidate))
            bot_coor = int(max(ver_candidate))
            width = right_coor - left_coor

            if target_shape is not None:
                scale_x = target_shape[0] / image_shape[0]
                scale_y = target_shape[1] / image_shape[1]
                left_coor *= scale_x
                right_coor *= scale_x
                top_coor *= scale_y
                bot_coor *= scale_y

            char_width = (width + num_char - 1) // num_char
            curr_left = left_coor
            for char_index in range(num_char):
                curr_right = curr_left + char_width

                seg_kv = segment["class"]
                if discard_key and seg_kv == "KEY":
                    char_class = 0
                else:
                    char_class = segment["tag"][char_index]

                char_pos_neg = 2 if (char_class == 0) else 1

                curr_row_dict = {
                    "left": [curr_left],
                    "top": [top_coor],
                    "right": [curr_right],
                    "bot": [bot_coor],
                    "text": [segment["string"][char_index]],
                    "data_class": [char_class],
                    "pos_neg": [char_pos_neg],
                }
                curr_row_dataframe = pd.DataFrame(curr_row_dict)
                csv_label = pd.concat(
                    [csv_label, curr_row_dataframe], axis=0, ignore_index=True
                )

                curr_left = curr_right

    csv_label.to_csv(dir_csv_label)


MODE_DICT = {
    # "full_seg": single_label_parser_full_seg,
    "ltp": single_label_parser_ltp,
    "char": single_label_parser_char,
    "char_BIO": single_label_parser_char_BIO,
}


def data_preprocessing_pipeline(
    root_dir_image: str,
    root_dir_json_label: str,
    root_dir_csv_label: str,
    target_shape: Tuple[int],
    mode: str,
    discard_key: bool,
) -> None:
    assert os.path.exists(
        root_dir_image
    ), f"The given image_root {root_dir_image} does not exists"
    assert os.path.exists(
        root_dir_json_label
    ), f"The given json label_root {root_dir_json_label} does not exists"
    if not os.path.exists(root_dir_csv_label):
        os.mkdir(root_dir_csv_label)

    assert mode in MODE_DICT.keys(), f"mode must be in {MODE_DICT.keys()}"

    file_list = os.listdir(root_dir_image)
    for file in tqdm(file_list):
        dir_image = os.path.join(root_dir_image, file)
        dir_json_label = os.path.join(root_dir_json_label, file.replace("jpg", "json"))
        dir_csv_label = os.path.join(root_dir_csv_label, file.replace("jpg", "csv"))
        MODE_DICT[mode](
            dir_img=dir_image,
            dir_json_label=dir_json_label,
            dir_csv_label=dir_csv_label,
            target_shape=target_shape,
            discard_key=discard_key,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--discard_key", type=bool)
    args = parser.parse_args()

    root = args.root

    image_root = os.path.join(root, "image")
    txt_label_root = os.path.join(root, "label")
    json_label_root = os.path.join(root, "_label_json")
    csv_label_root = os.path.join(root, "_label_csv")

    if not os.path.exists(json_label_root):
        generate_json(
            root_dir_txt_label=txt_label_root, root_dir_json_label=json_label_root
        )

    data_preprocessing_pipeline(
        root_dir_image=image_root,
        root_dir_json_label=json_label_root,
        root_dir_csv_label=csv_label_root,
        target_shape=None,
        mode=args.mode,
        discard_key=args.discard_key,
    )
