import os
import json
import tqdm
import argparse
import pandas as pd

from typing import Literal

FUNSD_CLASS_INDEX = {"other": 0, "question": 1, "answer": 2, "header": 3}


def annotation_parsing_word(dir_annotation: str, dir_save: str):
    with open(dir_annotation, "rb") as f:
        orig_annotation = json.load(f)

    csv_label = pd.DataFrame(
        columns=["left", "top", "right", "bot", "text", "data_class", "pos_neg"]
    )
    for seg in orig_annotation["form"]:
        data_class = seg["label"]
        pos_neg = 2 if data_class == 0 else 1
        for word in seg["words"]:
            word_text = word["text"]
            if len(word_text) == 0:
                continue
            if word_text == "N/A":
                word_text = Literal["N/A"]
            coors = word["box"]
            left = coors[0]
            top = coors[1]
            right = coors[2]
            bot = coors[3]

            curr_row_dict = {
                "left": [left],
                "top": [top],
                "right": [right],
                "bot": [bot],
                "text": [word_text],
                "data_class": [FUNSD_CLASS_INDEX[data_class]],
                "pos_neg": [pos_neg],
            }
            curr_row_dataframe = pd.DataFrame(curr_row_dict)
            csv_label = pd.concat(
                [csv_label, curr_row_dataframe], axis=0, ignore_index=True
            )

    csv_label.to_csv(dir_save.replace(".json", ".csv"))


def annotation_parsing_seg(dir_annotation: str, dir_save: str):
    with open(dir_annotation, "rb") as f:
        orig_annotation = json.load(f)

    csv_label = pd.DataFrame(
        columns=["left", "top", "right", "bot", "text", "data_class", "pos_neg"]
    )
    for seg in orig_annotation["form"]:
        seg_text = seg["text"]
        if len(seg_text) == 0:
            continue
        if seg_text == "N/A":
            seg_text = Literal["N/A"]
        if seg_text == "NA":
            seg_text = Literal["NA"]

        data_class = seg["label"]
        pos_neg = 2 if data_class == 0 else 1

        coors = seg["box"]
        left = coors[0]
        top = coors[1]
        right = coors[2]
        bot = coors[3]

        curr_row_dict = {
            "left": [left],
            "top": [top],
            "right": [right],
            "bot": [bot],
            "text": [seg_text],
            "data_class": [FUNSD_CLASS_INDEX[data_class]],
            "pos_neg": [pos_neg],
        }
        curr_row_dataframe = pd.DataFrame(curr_row_dict)
        csv_label = pd.concat(
            [csv_label, curr_row_dataframe], axis=0, ignore_index=True
        )

    csv_label.to_csv(dir_save.replace(".json", ".csv"))


DATA_LEVEL_DICT = {
    "word": annotation_parsing_word,
    "seg": annotation_parsing_seg,
}


def parse_multiple(dir_orig_root: str, dir_save_root: str, mode: str):
    assert mode in list(
        DATA_LEVEL_DICT.keys()
    ), f"invalid mode value {mode}, must be {DATA_LEVEL_DICT.keys()}"
    for file in tqdm.tqdm(os.listdir(dir_orig_root)):
        if not file.endswith(".json"):
            continue
        dir_annotation = os.path.join(dir_orig_root, file)
        dir_save = os.path.join(dir_save_root, file)
        DATA_LEVEL_DICT[mode](dir_annotation=dir_annotation, dir_save=dir_save)


def run_annotation_parser(dir_funsd_root: str, mode: str):
    dir_train_root = os.path.join(dir_funsd_root, "training_data", "annotations")
    dir_test_root = os.path.join(dir_funsd_root, "testing_data", "annotations")
    dir_train_save_root = os.path.join(dir_funsd_root, "training_data", "_label_csv")
    dir_test_save_root = os.path.join(dir_funsd_root, "testing_data", "_label_csv")
    if not os.path.exists(dir_train_save_root):
        os.mkdir(dir_train_save_root)
    if not os.path.exists(dir_test_save_root):
        os.mkdir(dir_test_save_root)

    parse_multiple(dir_train_root, dir_train_save_root, mode=mode)
    parse_multiple(dir_test_root, dir_test_save_root, mode=mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="dir to funsd root dir")
    parser.add_argument("--mode", type=str, help="label data level, word or seg")
    args = parser.parse_args()

    run_annotation_parser(args.root, args.mode)
