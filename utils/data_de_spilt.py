import os
from shutil import move, rmtree
from tqdm import tqdm

if __name__ == "__main__":
    data_root = r""

    dir_train_class = os.path.join(data_root, "train", "class")
    dir_train_image = os.path.join(data_root, "train", "image")
    dir_train_ocr_result = os.path.join(data_root, "train", "ocr_result")
    dir_train_pos_neg = os.path.join(data_root, "train", "pos_neg")

    dir_validate_class = os.path.join(data_root, "validate", "class")
    dir_validate_image = os.path.join(data_root, "validate", "image")
    dir_validate_ocr_result = os.path.join(data_root, "validate", "ocr_result")
    dir_validate_pos_neg = os.path.join(data_root, "validate", "pos_neg")

    val_file_list = os.listdir(os.path.join(data_root, "validate", "image"))
    for file in tqdm(val_file_list):
        src_class = os.path.join(dir_validate_class, file.replace("jpg", "npy"))
        src_image = os.path.join(dir_validate_image, file)
        src_ocr_result = os.path.join(
            dir_validate_ocr_result, file.replace("jpg", "csv")
        )
        src_pos_neg = os.path.join(dir_validate_pos_neg, file.replace("jpg", "npy"))

        des_class = os.path.join(dir_train_class, file.replace("jpg", "npy"))
        des_image = os.path.join(dir_train_image, file)
        des_ocr_result = os.path.join(dir_train_ocr_result, file.replace("jpg", "csv"))
        des_pos_neg = os.path.join(dir_train_pos_neg, file.replace("jpg", "npy"))

        move(src_class, des_class)
        move(src_image, des_image)
        move(src_ocr_result, des_ocr_result)
        move(src_pos_neg, des_pos_neg)

    rmtree(os.path.join(data_root, "validate"))
