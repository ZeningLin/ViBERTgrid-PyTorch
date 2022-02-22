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


def single_label_parser(
    dir_img: str,
    dir_json_label: str,
    dir_csv_label: str,
    dir_class: str,
    dir_pos_neg: str,
    target_shape: Tuple[int] = None,
):
    image = plt.imread(dir_img)
    image_shape = image.shape

    if target_shape is not None:
        assert (
            len(target_shape) == 2
        ), f"target_shape can only contain 2 elements, {len(target_shape)} given"
        pos_neg_label = np.zeros(target_shape, dtype=int)
        class_label = np.zeros(target_shape, dtype=int)
    else:
        pos_neg_label = np.zeros((image_shape[0], image_shape[1]), dtype=int)
        class_label = np.zeros((image_shape[0], image_shape[1]), dtype=int)

    csv_label = pd.DataFrame(
        columns=["left", "top", "right", "bot", "text", "data_class", "pos_neg"]
    )

    with open(dir_json_label, "r") as json_f:
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
                char_class = segment["tag"][char_index]
                char_pos_neg = 2 if (char_class == 0) else 1

                class_label[top_coor:bot_coor, curr_left:curr_right] = char_class
                pos_neg_label[top_coor:bot_coor, curr_left:curr_right] = char_pos_neg

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

    np.save(dir_class, class_label)
    np.save(dir_pos_neg, pos_neg_label)
    csv_label.to_csv(dir_csv_label)


def data_preprocessing_pipeline(
    root_dir_image: str,
    root_dir_json_label: str,
    root_dir_csv_label: str,
    root_dir_class: str,
    root_dir_pos_neg: str,
    target_shape: Tuple[int],
) -> None:
    assert os.path.exists(
        root_dir_image
    ), f"The given image_root {root_dir_image} does not exists"
    assert os.path.exists(
        root_dir_json_label
    ), f"The given json label_root {root_dir_json_label} does not exists"
    if not os.path.exists(root_dir_csv_label):
        os.mkdir(root_dir_csv_label)
    if not os.path.exists(root_dir_class):
        os.mkdir(root_dir_class)
    if not os.path.exists(root_dir_pos_neg):
        os.mkdir(root_dir_pos_neg)

    file_list = os.listdir(root_dir_image)
    for file in tqdm(file_list):
        dir_image = os.path.join(root_dir_image, file)
        dir_json_label = os.path.join(root_dir_json_label, file.replace("jpg", "json"))
        dir_csv_label = os.path.join(root_dir_csv_label, file.replace("jpg", "csv"))
        dir_class = os.path.join(root_dir_class, file.replace("jpg", "npy"))
        dir_pos_neg = os.path.join(root_dir_pos_neg, file.replace("jpg", "npy"))
        single_label_parser(
            dir_img=dir_image,
            dir_json_label=dir_json_label,
            dir_csv_label=dir_csv_label,
            dir_class=dir_class,
            dir_pos_neg=dir_pos_neg,
            target_shape=target_shape,
        )


if __name__ == "__main__":
    image_root = r"/home/zening_lin@intsig.com/文档/datasets//EPHOIE/image"
    txt_label_root = r"/home/zening_lin@intsig.com/文档/datasets//EPHOIE/label"
    json_label_root = r"/home/zening_lin@intsig.com/文档/datasets//EPHOIE/_label_json"
    csv_label_root = r"/home/zening_lin@intsig.com/文档/datasets//EPHOIE/_label_csv"
    class_label_root = r"/home/zening_lin@intsig.com/文档/datasets//EPHOIE/_class"
    pos_neg_label_root = r"/home/zening_lin@intsig.com/文档/datasets//EPHOIE/_pos_neg"

    if not os.path.exists(json_label_root):
        generate_json(
            root_dir_txt_label=txt_label_root, root_dir_json_label=json_label_root
        )

    data_preprocessing_pipeline(
        root_dir_image=image_root,
        root_dir_json_label=json_label_root,
        root_dir_csv_label=csv_label_root,
        root_dir_class=class_label_root,
        root_dir_pos_neg=pos_neg_label_root,
        target_shape=None,
    )
