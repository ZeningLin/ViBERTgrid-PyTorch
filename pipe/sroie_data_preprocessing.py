import os
import re
import json
import tqdm

import math
import scipy.sparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

import pytesseract

from typing import List, Tuple


def ocr_extraction(dir_image: str, conf_treshold: float = 10) -> pd.DataFrame:

    image_orig = plt.imread(dir_image, format='jpeg')
    information_dataframe = pytesseract.image_to_data(
        image_orig, output_type=pytesseract.Output.DATAFRAME)
    information_dataframe = information_dataframe[information_dataframe['conf'] > conf_treshold]
    information_dataframe = information_dataframe[[
        'left', 'top', 'width', 'height', 'text']]

    return information_dataframe
    pass


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
    pos_neg: int = 2
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
    return key_dataframe.append({
        'left': left,
        'right': right,
        'top': top,
        'bot': bot,
        'text': text,
        'data_class': data_class,
        'pos_neg': pos_neg
    }, ignore_index=True)


def ground_truth_extraction(
    dir_bbox: str,
    dir_key: str,
    data_classes: List,
    cosine_sim_treshold: float = 0.4
) -> pd.DataFrame:
    """extract ground truth information of the SROIE dataset

    Parameters
    ----------
    dir_bbox: str
        directory to the bbox csv file
    dir_key: str, 
        directory to the key info json file
    data_classes: List, 
        list of names of data classes
    cosine_sim_treshold: float = 0.4
        retrieval of key information uses cosine simularity
        this treshold controls the matching rate

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
    """
    with open(dir_bbox, 'r') as bbox_f:
        lines = bbox_f.readlines()
        gt_dataframe = pd.DataFrame(
            columns=['left', 'top', 'right', 'bot', 'text', 'data_class', 'pos_neg'])
        for line in lines:
            gt_all_info = line.split(',', maxsplit=8)
            left_coor = int(gt_all_info[0])
            top_coor = int(gt_all_info[1])
            right_coor = int(gt_all_info[4])
            bot_coor = int(gt_all_info[5])
            text = gt_all_info[-1]
            gt_dataframe = dataframe_append(
                gt_dataframe,
                left_coor,
                top_coor,
                right_coor,
                bot_coor,
                text
            )

    with open(dir_key, 'r') as key_f:
        key_info = json.load(key_f)
        for data_class in data_classes:
            if data_class not in key_info.keys():
                key_info[data_class] = 'UNKNOWN'
            key_info[data_class] = key_info[data_class].upper()

    count_vectorizer = CountVectorizer().fit_transform(
        [key_info[data_class]
            for data_class in data_classes] + list(gt_dataframe['text'])
    )

    total_float = re.search(r'([-+]?[0-9]*\.?[0-9]+)', key_info['total'])
    for index, row in gt_dataframe.iterrows():
        # retrieve 'company' in gt_dataframe
        if cosine_simularity(count_vectorizer[0].reshape(1, -1),
                             count_vectorizer[index + len(data_classes)].reshape(1, -1)) > cosine_sim_treshold:
            gt_dataframe.loc[index, 'data_class'] = 1
            gt_dataframe.loc[index, 'pos_neg'] = 1

        # retrieve 'address' in gt_dataframe
        if cosine_simularity(count_vectorizer[2].reshape(1, -1),
                             count_vectorizer[index + len(data_classes)].reshape(1, -1)) > cosine_sim_treshold:
            gt_dataframe.loc[index, 'data_class'] = 3
            gt_dataframe.loc[index, 'pos_neg'] = 1

        # retrieve 'date' in gt_dataframe
        tab_date = re.findall(
            r'((?i)(?:[12][0-9]|3[01]|0*[1-9])(?P<sep>[- \/.\\])(?P=sep)*(?:1[012]|0*[1-9]|jan(?:uary)?|feb('
            r'?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov('
            r'?:ember)?|dec(?:ember)?)(?P=sep)+(?:19|20)\d\d|(?:[12][0-9]|3[01]|0*[1-9])(?P<sep2>[- \/.\\])('
            r'?P=sep2)*(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul('
            r'?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P=sep2)+\d\d|(?:1[012]|0*['
            r'1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep('
            r'?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P<sep3>[- \/.\\])(?P=sep3)*(?:[12][0-9]|3[01]|0*['
            r'1-9])(?P=sep3)+(?:19|20)\d\d|(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr('
            r'?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)('
            r'?P<sep4>[- \/.\\])(?P=sep4)*(?:[12][0-9]|3[01]|0*[1-9])(?P=sep4)+\d\d|(?:19|20)\d\d(?P<sep5>[- \/.\\])('
            r'?P=sep5)*(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul('
            r'?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?P=sep5)+(?:[12][0-9]|3['
            r'01]|0*[1-9])|\d\d(?P<sep6>[- \/.\\])(?P=sep6)*(?:1[012]|0*[1-9]|jan(?:uary)?|feb(?:ruary)?|mar('
            r'?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec('
            r'?:ember)?)(?P=sep6)+(?:[12][0-9]|3[01]|0*[1-9])|(?:[12][0-9]|3[01]|0*[1-9])(?:jan(?:uary)?|feb('
            r'?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov('
            r'?:ember)?|dec(?:ember)?)(?:19|20)\d\d|(?:[12][0-9]|3[01]|0*[1-9])(?:jan(?:uary)?|feb(?:ruary)?|mar('
            r'?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec('
            r'?:ember)?)\d\d|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug('
            r'?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[12][0-9]|3[01]|0*[1-9])('
            r'?:19|20)\d\d|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug('
            r'?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[12][0-9]|3[01]|0*[1-9])\d\d|('
            r'?:19|20)\d\d(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug('
            r'?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[12][0-9]|3[01]|0*[1-9])|\d\d(?:jan('
            r'?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct('
            r'?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:[12][0-9]|3[01]|0*[1-9])|(?:[12][0-9]|3[01]|0[1-9])(?:1[012]|0['
            r'1-9])(?:19|20)\d\d|(?:1[012]|0[1-9])(?:[12][0-9]|3[01]|0[1-9])(?:19|20)\d\d|(?:19|20)\d\d(?:1[012]|0['
            r'1-9])(?:[12][0-9]|3[01]|0[1-9])|(?:1[012]|0[1-9])(?:[12][0-9]|3[01]|0[1-9])\d\d|(?:[12][0-9]|3[01]|0['
            r'1-9])(?:1[012]|0[1-9])\d\d|\d\d(?:1[012]|0[1-9])(?:[12][0-9]|3[01]|0[1-9]))',
            row['text'])
        for date in tab_date:
            if date[0] == key_info['date']:
                gt_dataframe.loc[index, 'data_class'] = 2
                gt_dataframe.loc[index, 'pos_neg'] = 1

        # retrieve 'total' in gt_dataframe
        tab_floats = re.findall(r'([-+]?[0-9]*\.?[0-9]+)', row["text"])
        if total_float:
            for float_ in tab_floats:
                if float(total_float.group(0)) == float(float_):
                    gt_dataframe.loc[index, 'data_class'] = 4
                    gt_dataframe.loc[index, 'pos_neg'] = 1

    return gt_dataframe


def generate_label(
    gt_dataframe: pd.DataFrame,
    img_shape: Tuple[int],
    num_class: int = 5,
    target_shape: Tuple[int] = None
) -> Tuple[np.ndarray]:
    """generate pos_neg labels (corresponds to X_1^{out} in sec 3.3 of the paper) and class labels
    (used for the "second classifier" in both head)

    Parameters
    ----------
    gt_dataframe : pandas.DataFrame
        ground_truth dataframe extracted by function ground_truth_extraction()
    img_shape : Tuple[int]
        shape of the original image
    num_class : int, optional
        number of classes, including backgound, by default 5
    target_shape : Tuple[int], optional
        shape of the reshaped image, reshape will be applied if not None, by default None

    Returns
    -------
    pos_neg_label: nunpy.ndarray
        pos_neg labels (corresponds to X_1^{out} in sec 3.3 of the paper)
    class_label: numpy.ndarray
        class labels (used for the "second classifier" in both head)
    """
    if(target_shape is not None):
        pos_neg_label = np.zeros(
            (3, target_shape[0], target_shape[1]), dtype=int)
        class_label = np.zeros(
            (2 * num_class, target_shape[0], target_shape[1]), dtype=int)
    else:
        pos_neg_label = np.zeros((3, img_shape[0], img_shape[1]), dtype=int)
        class_label = np.zeros(
            (2 * num_class, img_shape[0], img_shape[1]), dtype=int)
    for _, row in gt_dataframe.iterrows():
        left_coor = row['left']
        right_coor = row['right']
        top_coor = row['top']
        bot_coor = row['bot']
        pos_neg = row['pos_neg']
        data_class = row['data_class']

        if(target_shape is not None):
            left_coor = int((left_coor / img_shape[0]) * target_shape[0])
            right_coor = int((right_coor / img_shape[0]) * target_shape[0])
            top_coor = int((top_coor / img_shape[1]) * target_shape[1])
            bot_coor = int((bot_coor / img_shape[1]) * target_shape[1])

        pos_neg_label[pos_neg, top_coor:bot_coor, left_coor:right_coor] = 1
        class_label[2 * data_class, top_coor:bot_coor,
                    left_coor:right_coor] = 1

    return pos_neg_label, class_label


def train_data_preprocessing_pipeline(
    data_classes: List[str],
    dir_train_root: str,
    dir_processed: str,
    cosine_sim_treshold: float = 0.4,
    target_shape: Tuple[int] = None
) -> None:
    """extract ground-truth information and generate labels in ViBERTgrid's format

    Parameters
    ----------
    data_classes : List[str]
        list of data classes
    dir_train_root : str
        root of train data
    dir_processed : str
        root of labels
    cosine_sim_treshold : float, optional
        cosine simularity treshold used in retrieval, by default 0.4
    target_shape : Tuple[int], optional
        shape of the reshaped image, reshape will be applied if not None, by default None
    """
    num_classes = len(data_classes) + 1

    dir_train_img = os.path.join(dir_train_root, 'img')
    dir_train_bbox = os.path.join(dir_train_root, 'box')
    dir_train_key = os.path.join(dir_train_root, 'key')

    dir_pos_neg = os.path.join(dir_processed, 'pos_neg')
    if not os.path.exists(dir_pos_neg):
        os.mkdir(dir_pos_neg)
    dir_class = os.path.join(dir_processed, 'class')
    if not os.path.exists(dir_class):
        os.mkdir(dir_class)

    train_list = [f for f in os.listdir(dir_train_img)]
    print("preprocessing dataset")
    for file in tqdm.tqdm(train_list):
        dir_image = os.path.join(dir_train_img, file)
        dir_bbox = os.path.join(dir_train_bbox, file.replace('jpg', 'csv'))
        dir_key = os.path.join(dir_train_key, file.replace('jpg', 'json'))
        dir_pos_neg_ = os.path.join(dir_pos_neg, file.replace('jpg', 'npy'))
        dir_class_ = os.path.join(dir_class, file.replace('jpg', 'npy'))

        img = plt.imread(dir_image)
        img_shape = img.shape[0:2]

        gt_dataframe = ground_truth_extraction(
            dir_bbox=dir_bbox,
            dir_key=dir_key,
            data_classes=data_classes,
            cosine_sim_treshold=cosine_sim_treshold
        )
        pos_neg_label, class_label = generate_label(
            gt_dataframe=gt_dataframe,
            img_shape=img_shape,
            num_class=num_classes,
            target_shape=target_shape
        )

        np.save(dir_pos_neg_, pos_neg_label)
        np.save(dir_class_, class_label)


if __name__ == '__main__':
    target_shape = (336, 256)
    data_classes = ['company', 'date', 'address', 'total']
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    dir_train_root = r'D:\PostGraduate\DataSet\ICDAR-SROIE\train_raw'
    dir_processed = r'D:\PostGraduate\DataSet\ICDAR-SROIE\ViBERTgrid_format'

    train_data_preprocessing_pipeline(
        data_classes=data_classes,
        dir_train_root=dir_train_root,
        dir_processed=dir_processed,
        target_shape=target_shape)
