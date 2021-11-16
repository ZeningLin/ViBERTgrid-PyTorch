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


RESIZE_SHAPE = (336, 256)


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
            torchvision.transforms.Resize(
                RESIZE_SHAPE,
                interpolation=torchvision.transforms.InterpolationMode.NEAREST
            ),
            torchvision.transforms.ToTensor()
        ])

        if self.train:
            self.img_list, \
                self.class_list, \
                self.pos_neg_list, \
                self.ocr_result_list, = self._extract_train(self.root)

    def __len__(self) -> int:
        return len(self.img_list)

    # @pysnooper.snoop()
    def __getitem__(self, index):
        if self.train:
            return_ocr_result = self.ocr_result_list[index]
            ocr_coor = []
            ocr_text = []
            for item in return_ocr_result:
                ocr_coor.append(list(map(int, item[:4])))
                curr_text = item[4:]
                curr_text = ''.join(curr_text)
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

            return (self.transform_img(self.img_list[index]),
                    torch.tensor(self.class_list[index]),
                    torch.tensor(self.pos_neg_list[index]),
                    torch.tensor(ocr_coor_expand, dtype=torch.long),
                    torch.tensor(ocr_corpus, dtype=torch.long))

    # @pysnooper.snoop()
    def _ViBERTgrid_coll_func(self, samples):
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

        return (torch.stack(imgs, dim=0),
                torch.stack(class_labels, dim=0),
                torch.stack(pos_neg_labels, dim=0),
                ocr_coors,
                ocr_corpus,
                mask)

    def _extract_train(self, root: str):
        dir_img = os.path.join(root, 'img')
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
                ocr_result_list.append([data_line.strip().split(',')[
                                       1:] for data_line in data_lines])

        return img_list, np.array(class_list), np.array(pos_neg_list), ocr_result_list


def load_train_dataset(
    root: str,
    batch_size: int,
    val_ratio: float = 0.3,
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
    val_ratio : float, optional
        ratio of validation subset, by default 0.3
    num_workers : int, optional
        number of workers in dataloader, by default 0
    tokenizer : optional
        tokenizer

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader

    """
    SROIE_dataset = SROIEDataset(root, train=True, tokenizer=tokenizer)
    len_val = int(len(SROIE_dataset) * val_ratio)
    len_train = len(SROIE_dataset) - len_val
    train_dataset, val_dataset = random_split(
        SROIE_dataset, [len_train, len_val])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=SROIE_dataset._ViBERTgrid_coll_func)

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=SROIE_dataset._ViBERTgrid_coll_func)

    return train_loader, val_loader


if __name__ == '__main__':
    dir_processed = r'D:\PostGraduate\DataSet\ICDAR-SROIE\ViBERTgrid_format\train'
    model_version = 'bert-base-uncased'
    print('loading bert pretrained')
    tokenizer = BertTokenizer.from_pretrained(model_version)
    train_loader, val_loader = load_train_dataset(
        dir_processed,
        batch_size=4,
        val_ratio=0.3,
        num_workers=0,
        tokenizer=tokenizer
    )

    train_batch = next(iter(train_loader))
    print(train_batch)
