import os
import random
from shutil import move
from tqdm import tqdm

if __name__ == "__main__":
    validate_ratio = 0.3
    data_root = r"dir_to_data_root"  # change to you directory

    if not os.path.exists(data_root):
        raise ValueError("data not found, please check your root")

    if not os.path.exists(os.path.join(data_root, "validate")):
        os.mkdir(os.path.join(data_root, "validate"))

    if not os.path.exists(os.path.join(data_root, "validate", "image")):
        os.mkdir(os.path.join(data_root, "validate", "image"))
    if not os.path.exists(os.path.join(data_root, "validate", "class")):
        os.mkdir(os.path.join(data_root, "validate", "class"))
    if not os.path.exists(os.path.join(data_root, "validate", "ocr_result")):
        os.mkdir(os.path.join(data_root, "validate", "ocr_result"))
    if not os.path.exists(os.path.join(data_root, "validate", "pos_neg")):
        os.mkdir(os.path.join(data_root, "validate", "pos_neg"))

    file_list = os.listdir(os.path.join(data_root, "train", "image"))
    file_count = len(file_list)
    num_val = int(file_count * validate_ratio)
    val_index = random.sample(range(file_count), num_val)
    for index in tqdm(val_index):
        file_name = file_list[index]
        src_image = os.path.join(data_root, "train", "image", file_name)
        src_class = os.path.join(
            data_root, "train", "class", file_name.replace("jpg", "npy")
        )
        src_ocr_result = os.path.join(
            data_root, "train", "ocr_result", file_name.replace("jpg", "csv")
        )
        src_pos_neg = os.path.join(
            data_root, "train", "pos_neg", file_name.replace("jpg", "npy")
        )

        des_image = os.path.join(data_root, "validate", "image", file_name)
        des_class = os.path.join(
            data_root, "validate", "class", file_name.replace("jpg", "npy")
        )
        des_ocr_result = os.path.join(
            data_root, "validate", "ocr_result", file_name.replace("jpg", "csv")
        )
        des_pos_neg = os.path.join(
            data_root, "validate", "pos_neg", file_name.replace("jpg", "npy")
        )

        move(src_image, des_image)
        move(src_class, des_class)
        move(src_ocr_result, des_ocr_result)
        move(src_pos_neg, des_pos_neg)
