import os
# import pysnooper
from tqdm import tqdm
from typing import Tuple, Optional, Callable

import numpy as np
import PIL.Image as Image

import torch
import torchvision
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer


class SROIEDataset(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        tokenizer: Optional[Callable] = None
    ) -> None:
        super().__init__()

        assert os.path.exists(root), "the given root path does not exists"
        assert tokenizer is not None, "no tokenizer given"

        self.root = root
        self.train = train
        self.tokenizer = tokenizer
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', add_special_tokens = True)
        self.max_length = 0
        self.transform_img = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

        dir_img = os.path.join(root, 'image')
        self.filename_list = [f for f in os.listdir(dir_img)]

    def __len__(self) -> int:
        return len(self.filename_list)

    def __getitem__(self, index):
        dir_img = os.path.join(self.root, 'image')
        dir_class = os.path.join(self.root, 'class')
        dir_pos_neg = os.path.join(self.root, 'pos_neg')
        dir_ocr_result = os.path.join(self.root, 'ocr_result')

        file = self.filename_list[index]

        image = Image.open(os.path.join(dir_img, file))

        data_class = np.load(os.path.join(
            dir_class, file.replace('jpg', 'npy')))

        pos_neg = np.load(os.path.join(
            dir_pos_neg, file.replace('jpg', 'npy')))

        with open(os.path.join(dir_ocr_result, file.replace('jpg', 'csv')), 'r', encoding='utf-8') as csv_file:
            data_lines = csv_file.readlines()[1:]
            self.max_length = len(data_lines) if (
                len(data_lines) > self.max_length) else self.max_length
            # # debug----------------------------------------------------
            # curr_ocr_result = [data_line.strip().split(',')[1:]
            #                     for data_line in data_lines]
            # if(len(curr_ocr_result) == 0):
            #     print('\nempty result found in data extraction, please check')
            #     print(file.replace('.jpg', 'csv'))
            # # ---------------------------------------------------------
            ocr_result = [data_line[:-2].strip().split(',')[1:]
                          for data_line in data_lines]

        ocr_coor = []
        ocr_text = []
        for item in ocr_result:
            curr_text = item[4:]
            # avoid " ' , in ocr recognition results affect list splitting
            curr_text = ''.join(curr_text)
            curr_text.replace('\"', '')
            if curr_text == '':
                continue        # discard empty results
            ocr_coor.append(list(map(int, item[:4])))
            ocr_text.append(curr_text)

        ocr_coor_expand = []
        ocr_tokens = []
        for text, coor in zip(ocr_text, ocr_coor):
            if text == '':
                continue
            curr_tokens = self.tokenizer.tokenize(text)
            for i in range(len(curr_tokens)):
                ocr_coor_expand.append(coor)
                ocr_tokens.append(curr_tokens[i])

        ocr_corpus = self.tokenizer.convert_tokens_to_ids(ocr_tokens)

        return (self.transform_img(image),
                torch.tensor(data_class),
                torch.tensor(pos_neg),
                torch.tensor(ocr_coor_expand, dtype=torch.long),
                torch.tensor(ocr_corpus, dtype=torch.long))

    @staticmethod
    def _coll_func(batch):
        return tuple(zip(*batch))

    @staticmethod
    def _ViBERTgrid_coll_func(samples):
        imgs = []
        class_labels = []
        pos_neg_labels = []
        ocr_coors = []
        ocr_corpus = []
        for item in samples:
            imgs.append(item[0])
            class_labels.append(item[1])
            pos_neg_labels.append(item[2])
            ocr_coors.append(item[3])
            ocr_corpus.append(item[4])

        # pad sequence to generate mini-batch
        ocr_coors = pad_sequence(ocr_coors, batch_first=True)
        ocr_corpus = pad_sequence(ocr_corpus, batch_first=True)
        # add mask to indicate valid corpus
        mask = torch.zeros(ocr_corpus.shape, dtype=torch.long)
        mask = mask.masked_fill_((ocr_corpus != 0), 1)

        return (tuple(imgs),
                torch.stack(class_labels, dim=0),
                torch.stack(pos_neg_labels, dim=0),
                ocr_coors,
                ocr_corpus,
                mask)

    def _extract_train(self, root: str):
        dir_img = os.path.join(root, 'image')
        dir_class = os.path.join(root, 'class')
        dir_pos_neg = os.path.join(root, 'pos_neg')
        dir_ocr_result = os.path.join(root, 'ocr_result')

        filename_list = [f for f in os.listdir(dir_img)]

        img_list = []
        ocr_result_list = []
        class_list = []
        pos_neg_list = []

        print("extracting SROIE train data")
        for file in tqdm(filename_list):
            file: str

            image = Image.open(os.path.join(dir_img, file))
            img_list.append(image)

            data_class = np.load(os.path.join(
                dir_class, file.replace('jpg', 'npy')))
            class_list.append(data_class)

            pos_neg = np.load(os.path.join(
                dir_pos_neg, file.replace('jpg', 'npy')))
            pos_neg_list.append(pos_neg)

            with open(os.path.join(dir_ocr_result, file.replace('jpg', 'csv')), 'r', encoding='utf-8') as csv_file:
                data_lines = csv_file.readlines()[1:]
                self.max_length = len(data_lines) if (
                    len(data_lines) > self.max_length) else self.max_length
                # debug----------------------------------------------------
                curr_ocr_result = [data_line.strip().split(',')[1:]
                                   for data_line in data_lines]
                if(len(curr_ocr_result) == 0):
                    print('\nempty result found in data extraction, please check')
                    print(file.replace('.jpg', 'csv'))
                # ---------------------------------------------------------
                ocr_result_list.append([data_line[:-2].strip().split(',')[
                    1:] for data_line in data_lines])

        return img_list, np.array(class_list), np.array(pos_neg_list), ocr_result_list


def load_train_dataset(
    root: str,
    batch_size: int,
    num_workers: int = 0,
    tokenizer: Optional[Callable] = None
) -> Tuple[DataLoader]:
    """load SROIE trainset then spilt into train & validation subset

    Parameters
    ----------
    root : str
        root of dataset
    batch_size : int
        batch size
    num_workers : int, optional
        number of workers in dataloader, by default 0
    tokenizer : optional
        tokenizer

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader

    """
    SROIE_train_dataset = SROIEDataset(root, train=True, tokenizer=tokenizer)
    SROIE_val_dataset = SROIEDataset(root, train=False, tokenizer=tokenizer)

    train_loader = DataLoader(
        SROIE_train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=SROIE_train_dataset._ViBERTgrid_coll_func)

    val_loader = DataLoader(
        SROIE_val_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=SROIE_val_dataset._ViBERTgrid_coll_func)

    return train_loader, val_loader


if __name__ == '__main__':
    dir_processed = r'D:\PostGraduate\DataSet\ICDAR-SROIE\ViBERTgrid_format\train'
    model_version = 'bert-base-uncased'
    print('loading bert pretrained')
    tokenizer = BertTokenizer.from_pretrained(model_version)
    train_loader, val_loader = load_train_dataset(
        dir_processed,
        batch_size=4,
        num_workers=0,
        tokenizer=tokenizer
    )

    for train_batch in tqdm(train_loader):
        img, class_label, pos_neg, coor, corpus, mask = train_batch
        for item in coor:
            if len(item) == 0:
                print('empty item found')
                print(item)
