import os
import re
import json
import tqdm
import argparse
import multiprocessing

import math
import scipy.sparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

from typing import List, Tuple

LOG_DIR = "./log.txt"


def cosine_simularity(a: scipy.sparse.csr_matrix, b: scipy.sparse.csr_matrix) -> float:
    """calculate cosine simularity of two given np.array

    Parameters
    ----------
    a, b : scipy.sparse.csr_matrix

    Returns
    -------
    cosine_simularity: float

    """
    norm_a = 0
    norm_b = 0
    a_dot_b = 0
    for i in range(a.nnz):
        index_a = a.indices[i]
        data_a = a.data[i]
        norm_a += data_a
        for j in range(b.nnz):
            index_b = b.indices[j]
            data_b = b.data[j]
            if i == 0:
                norm_b += data_b
            if index_a == index_b:
                a_dot_b += data_a * data_b
    return a_dot_b / (math.sqrt(norm_a * norm_b) + 1e-8)


def dataframe_append(
    key_dataframe: pd.DataFrame,
    left: int,
    top: int,
    right: int,
    bot: int,
    text: str,
    data_class: int = 0,
    pos_neg: int = 2,
) -> pd.DataFrame:
    """append a row to the given dataframe, with class information

    Parameters
    ----------
    key_dataframe: pandas.DataFrame
        dataframe that will be appended to
    left: int, top: int, right: int, bot: int,
        coor of bbox
    text: str
        text content inside bbox
    data_class: int = 0
        data_class of the appended row
    pos_neg: int = 2
        pos_neg flag, corresponds to X_1^{out} in sec 3.3 of the paper

    Returns
    -------
    appended_dataframe: pandas.DataFrame
        the processed ground_truth_dataframe

    """
    return key_dataframe.append(
        {
            "left": left,
            "right": right,
            "top": top,
            "bot": bot,
            "text": text,
            "data_class": data_class,
            "pos_neg": pos_neg,
        },
        ignore_index=True,
    )


def ground_truth_extraction(
    dir_img: str,
    dir_bbox: str,
    dir_key: str,
    data_classes: List,
    cosine_sim_treshold: float = 0.4,
    spilt_word: bool = False,
    target_shape: Tuple[int] = None,
) -> Tuple[pd.DataFrame, Tuple[int]]:
    """extract ground truth information of the SROIE dataset

    Parameters
    ----------
    dir_img: str
        directory to the image file
    dir_bbox: str
        directory to the bbox txt file
    dir_key: str,
        directory to the key info json file
    data_classes: List,
        list of names of data classes
    cosine_sim_treshold: float = 0.4
        retrieval of key information uses cosine simularity
        this treshold controls the matching rate
    spilt_word: bool, optional
        spilt gt labels into word-level, by default true

    Returns
    ------
    gt_dataframe: pandas.Dataframe
        dataframe that contains the following information
        - coordinates 'left', 'top', 'right', 'bot'
        - text
        - data_class
        - pos_neg

        where:
            data_class_value
                0:bkg
                1:company
                2:date
                3:address
                4:total
            pos_neg_value
                0: not inside word-box
                1: inside pre-defined word-box
                2: inside other box
    image_shape: Tuple[int]
        image shape
    """
    image = plt.imread(dir_img)
    image_shape = image.shape

    with open(dir_bbox, "r", encoding="utf-8") as bbox_f:
        lines = bbox_f.readlines()
        gt_dataframe = pd.DataFrame(
            columns=["left", "top", "right", "bot", "text", "data_class", "pos_neg"]
        )
        for line in lines:
            gt_all_info = line.split(",", maxsplit=8)

            if len(gt_all_info) < 8:
                continue  # discard invalid lines like '\n' in the labels

            left_coor = int(gt_all_info[0])
            top_coor = int(gt_all_info[1])
            right_coor = int(gt_all_info[4])
            bot_coor = int(gt_all_info[5])
            text = gt_all_info[8:]
            text = "".join(text)
            text = text.replace("\n", "")

            if spilt_word:
                # spilt text lines into word level
                text_words = text.split(" ")
                total_length = len(text)
                char_length = (right_coor - left_coor) / total_length
                edge_ptr = left_coor
                for text_word in text_words:
                    word_length = len(text_word)

                    left_coor_ = edge_ptr
                    top_coor_ = top_coor
                    right_coor_ = int(edge_ptr + word_length * char_length)
                    bot_coor_ = bot_coor

                    if target_shape is not None:
                        left_coor_ = int(
                            (left_coor_ / image_shape[1]) * target_shape[1]
                        )
                        top_coor_ = int((top_coor_ / image_shape[0]) * target_shape[0])
                        right_coor_ = int(
                            (right_coor_ / image_shape[1]) * target_shape[1]
                        )
                        bot_coor_ = int((bot_coor_ / image_shape[0]) * target_shape[0])

                    gt_dataframe = dataframe_append(
                        gt_dataframe,
                        left_coor_,
                        top_coor_,
                        right_coor_,
                        bot_coor_,
                        text_word,
                    )

                    edge_ptr += int((word_length + 1) * char_length)
            else:
                if target_shape is not None:
                    left_coor = int((left_coor / image_shape[1]) * target_shape[1])
                    top_coor = int((top_coor / image_shape[0]) * target_shape[0])
                    right_coor = int((right_coor / image_shape[1]) * target_shape[1])
                    bot_coor = int((bot_coor / image_shape[0]) * target_shape[0])

                gt_dataframe = dataframe_append(
                    gt_dataframe, left_coor, top_coor, right_coor, bot_coor, text
                )

    with open(dir_key, "r", encoding="utf-8") as key_f:
        key_info = json.load(key_f)
        for data_class in data_classes:
            if data_class not in key_info.keys():
                key_info[data_class] = "UNKNOWN"
            key_info[data_class] = key_info[data_class].upper()

    count_vectorizer = CountVectorizer().fit_transform(
        [key_info[data_class] for data_class in data_classes]
        + list(gt_dataframe["text"])
    )

    total_float = re.search(r"([-+]?[0-9]*\.?[0-9]+)", key_info["total"])
    for index, row in gt_dataframe.iterrows():
        # default value
        gt_dataframe.loc[index, "pos_neg"] = 2

        # retrieve 'company' in gt_dataframe
        if (
            cosine_simularity(
                count_vectorizer[0].reshape(1, -1),
                count_vectorizer[index + len(data_classes)].reshape(1, -1),
            )
            > cosine_sim_treshold
        ):
            gt_dataframe.loc[index, "data_class"] = 1
            gt_dataframe.loc[index, "pos_neg"] = 1

        # retrieve 'address' in gt_dataframe
        if (
            cosine_simularity(
                count_vectorizer[2].reshape(1, -1),
                count_vectorizer[index + len(data_classes)].reshape(1, -1),
            )
            > cosine_sim_treshold
        ):
            gt_dataframe.loc[index, "data_class"] = 3
            gt_dataframe.loc[index, "pos_neg"] = 1

        # retrieve 'date' in gt_dataframe
        tab_date = re.findall(
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
            r"1-9])(?:1[012]|0[1-9])\d\d|\d\d(?:1[012]|0[1-9])(?:[12][0-9]|3[01]|0[1-9]))",
            row["text"],
        )
        for date in tab_date:
            if date[0] == key_info["date"]:
                gt_dataframe.loc[index, "data_class"] = 2
                gt_dataframe.loc[index, "pos_neg"] = 1

        # retrieve 'total' in gt_dataframe
        tab_floats = re.findall(r"([-+]?[0-9]*\.?[0-9]+)", row["text"])
        if total_float:
            for float_ in tab_floats:
                if float(total_float.group(0)) == float(float_):
                    gt_dataframe.loc[index, "data_class"] = 4
                    gt_dataframe.loc[index, "pos_neg"] = 1

    return gt_dataframe, image_shape


def data_preprocessing_pipeline(
    dir_data_img: str,
    dir_data_bbox: str,
    dir_data_key: str,
    dir_ocr_result: str,
    file_list: List,
    data_classes: Tuple[int],
    cosine_sim_treshold: float = 0.4,
    spilt_word: bool = True,
    target_shape: Tuple[int] = None,
) -> None:
    """extract ground-truth information and generate labels in ViBERTgrid's format

    Parameters
    ----------
    dir_data_img : str
        directory of data image
    dir_data_bbox : str
        directory of ground-turth bbox txt files
    dir_data_key : str
        directory of key information json files
    dir_ocr_result : str
        directory of image ocr result
    file_list : List
        list of files to be processed
    num_classes : int, optional
        number of key information classes, including background, by default 5
    cosine_sim_treshold : float, optional
        cosine simularity treshold for key information retrieval in bbox labels, by default 0.4
    spilt_word: bool, optional
        spilt gt labels into word-level, by default true
    target_shape : Tuple[int], optional
        target shape of resized image, reshape will be applied to the original image if given, by default None
    """
    num_classes = len(data_classes) + 1
    for file in tqdm.tqdm(file_list):
        dir_image = os.path.join(dir_data_img, file)
        dir_bbox = os.path.join(dir_data_bbox, file.replace("jpg", "txt"))
        dir_key = os.path.join(dir_data_key, file.replace("jpg", "txt"))
        dir_ocr_result_ = os.path.join(dir_ocr_result, file.replace("jpg", "csv"))

        gt_dataframe, img_shape = ground_truth_extraction(
            dir_img=dir_image,
            dir_bbox=dir_bbox,
            dir_key=dir_key,
            data_classes=data_classes,
            cosine_sim_treshold=cosine_sim_treshold,
            spilt_word=spilt_word,
            target_shape=target_shape,
        )

        gt_dataframe.to_csv(dir_ocr_result_)


def data_parser(
    data_classes: List[str],
    dir_data_root: str,
    dir_processed: str,
    spilt_word: bool = True,
    cosine_sim_treshold: float = 0.4,
    target_shape: Tuple[int] = None,
):
    """pipeline for extracting ground-truth information
       and generate labels in ViBERTgrid's format

    Parameters
    ----------
    data_classes : List[str]
        list of data classes
    dir_data_root : str
        root of data
    dir_processed : str
        root of labels
    spilt_word: bool, optional
        spilt gt labels into word-level, by default true
    cosine_sim_treshold : float, optional
        cosine simularity treshold used in retrieval, by default 0.4
    target_shape : Tuple[int], optional
        shape of the reshaped image, reshape will be applied if not None, by default None
    """
    dir_data_img = os.path.join(dir_data_root, "img")
    dir_data_bbox = os.path.join(dir_data_root, "box")
    dir_data_key = os.path.join(dir_data_root, "key")

    dir_ocr_result = os.path.join(dir_processed, "ocr_result")
    if not os.path.exists(dir_ocr_result):
        os.mkdir(dir_ocr_result)

    print("preprocessing dataset in dir {}".format(dir_data_root))

    data_list = [f for f in os.listdir(dir_data_img)]
    data_preprocessing_pipeline(
        dir_data_img=dir_data_img,
        dir_data_bbox=dir_data_bbox,
        dir_data_key=dir_data_key,
        dir_ocr_result=dir_ocr_result,
        file_list=data_list,
        data_classes=data_classes,
        spilt_word=spilt_word,
        cosine_sim_treshold=cosine_sim_treshold,
        target_shape=target_shape,
    )

    print("process finished")


def data_parser_multiprocessing(
    data_classes: List[str],
    dir_data_root: str,
    dir_processed: str,
    spilt_word: bool = True,
    cosine_sim_treshold: float = 0.4,
    target_shape: Tuple[int] = None,
):
    """a multiprocessing pipeline for extracting ground-truth information
       and generate labels in ViBERTgrid's format
       METHOD NOT RECOMMENDED

    Parameters
    ----------
    data_classes : List[str]
        list of data classes
    dir_data_root : str
        root of data data
    dir_processed : str
        root of labels
    spilt_word: bool, optional
        spilt gt labels into word-level, by default true
    cosine_sim_treshold : float, optional
        cosine simularity treshold used in retrieval, by default 0.4
    target_shape : Tuple[int], optional
        shape of the reshaped image, reshape will be applied if not None, by default None
    """
    dir_data_img = os.path.join(dir_data_root, "img")
    dir_data_bbox = os.path.join(dir_data_root, "box")
    dir_data_key = os.path.join(dir_data_root, "key")

    dir_ocr_result = os.path.join(dir_processed, "ocr_result")
    if not os.path.exists(dir_ocr_result):
        os.mkdir(dir_ocr_result)
    dir_pos_neg = os.path.join(dir_processed, "pos_neg")
    if not os.path.exists(dir_pos_neg):
        os.mkdir(dir_pos_neg)
    dir_class = os.path.join(dir_processed, "class")
    if not os.path.exists(dir_class):
        os.mkdir(dir_class)

    data_list = [f for f in os.listdir(dir_data_img)]

    num_worker = os.cpu_count() // 2
    step_length = len(data_list) // num_worker
    processes = []
    start = 0

    print("preprocessing dataset")

    for i in range(num_worker):
        end = (i + 1) * step_length
        curr_data_list = (
            data_list[start:end] if end < len(data_list) else data_list[start:]
        )
        if len(curr_data_list) > 0:
            process = multiprocessing.Process(
                target=data_preprocessing_pipeline,
                args=(
                    dir_data_img,
                    dir_data_bbox,
                    dir_data_key,
                    dir_ocr_result,
                    dir_pos_neg,
                    dir_class,
                    curr_data_list,
                    data_classes,
                    spilt_word,
                    cosine_sim_treshold,
                    target_shape,
                ),
            )
            processes.append(process)

    for process in processes:
        process: multiprocessing.Process
        process.start()
        print("process {} started".format(process.pid))

    for process in processes:
        process: multiprocessing.Process
        process.join()
        print("process {} finished".format(process.pid))

    print("data preprocessing finished")


if __name__ == "__main__":
    """
    ___train_raw
      |___img: images
      |___box: txt files that contain OCR results
      |___key: txt files that contain key info labels

    ___test_raw
      |___img: images
      |___box: txt files that contain OCR results
      |___key: txt files that contain key info labels

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_raw", type=str, help="dir to train raw data root")
    parser.add_argument("--train_processed", type=str, help="dir to train processed")
    parser.add_argument("--test_raw", type=str, help="dir to test raw data root")
    parser.add_argument("--test_processed", type=str, help="dir to test processed")
    args = parser.parse_args()

    data_classes = ["company", "date", "address", "total"]

    dir_train_root = args.train_raw
    dir_processed = args.train_processed

    if not os.path.exists(dir_processed):
        os.mkdir(dir_processed)

    data_parser(
        data_classes=data_classes,
        dir_data_root=dir_train_root,
        dir_processed=dir_processed,
        target_shape=None,
    )

    dir_test_root = args.test_raw
    dir_processed = args.test_processed

    if not os.path.exists(dir_processed):
        os.mkdir(dir_processed)

    data_parser(
        data_classes=data_classes,
        dir_data_root=dir_test_root,
        dir_processed=dir_processed,
        target_shape=None,
    )
