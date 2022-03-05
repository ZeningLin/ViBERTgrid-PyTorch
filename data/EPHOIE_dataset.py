import os

from tqdm import tqdm
from typing import Tuple, Optional, Callable

import numpy as np
import pandas as pd
import PIL.Image as Image

import torch
import torchvision
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import distributed, Dataset, DataLoader, BatchSampler
from transformers import BertTokenizer


class EPHOIEDataset(Dataset):
    """The EPHOIE dataset
       
       can be downloaded from https://github.com/HCIILAB/EPHOIE after application


    Parameters
    ----------
    root : str
        root directory of the dataset, which contains 'train' and 'validate' folders
    train : bool, optional
        set True if loading training set, else False, by default True
    tokenizer : Optional[Callable], optional
        tokenizer used for corpus generation,
        from huggingface/transformer library,
        e.g transformer.BertTokenizer,
        by default None

    Returns
    -------
    imgs: tuple[torch.Tensor]
        tuple of original images,
        each with shape [3, H, W]
        need further transforms for forward propagation
    class_labels: tuple[torch.Tensor]
        tuple of class labels, same in shape with images,
        each with shape [num_classes, H, W].
        if the pixel at (x, y) belongs to class [3], then the value at
        channel [3] (in other words, at coor (3, x, y)) is one and zero
        at other channels.
        need further transforms for forward propagation
    pos_neg_labels: tuple[torch.Tensor]
        tuple of pos_neg labels, same in shape with images,
        each with shape [3, H, W].
        channel [0] = 1 if the pixel belongs to background
        channel [1] = 1 if the pixel belongs to key text region
        channel [2] = 1 if the pixel belongs to non-key text region
        need further transforms for forward propagation
    ocr_coors: torch.Tensor
        coordinates of each token
    ocr_corpus: torch.Tensor
        corpus generated from the given OCR result, padding performed.
    mask: torch.Tensor
        BERT-like model requires input with constant length (typically 512),
        if len(corpus) < constant_length, padding will be performed.
        mask indicates where the padding steps are. len(mask) = constant_length,
        mask[step] = 0 at padding steps, 1 otherwise.

    """

    def __init__(
        self, root: str, train: bool, tokenizer: Optional[Callable] = None
    ) -> None:
        super().__init__()

        assert os.path.exists(root), f"the given root path {root} does not exists"
        assert tokenizer is not None, "no tokenizer given"

        self.root = root
        self.train = train
        self.tokenizer = tokenizer
        self.max_length = 0
        self.transform_img = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )

        self.filename_list = []
        if self.train:
            with open(os.path.join(root, "train.txt"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    self.filename_list.append(line.strip('\n'))
        else:
            with open(os.path.join(root, "test.txt"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    self.filename_list.append(line.strip('\n'))

    def __len__(self) -> int:
        return len(self.filename_list)

    def __getitem__(self, index):
        dir_img = os.path.join(self.root, "image", (self.filename_list[index] + ".jpg"))
        dir_class = os.path.join(
            self.root, "_class", (self.filename_list[index] + ".npy")
        )
        dir_pos_neg = os.path.join(
            self.root, "_pos_neg", (self.filename_list[index] + ".npy")
        )
        dir_csv_label = os.path.join(
            self.root, "_label_csv", (self.filename_list[index] + ".csv")
        )

        image = Image.open(dir_img)
        if len(image.split()) != 3:
            image = image.convert("RGB")
        data_class = np.load(dir_class)
        pos_neg = np.load(dir_pos_neg)

        ocr_coor = []
        ocr_text = []
        csv_label: pd.DataFrame = pd.read_csv(dir_csv_label)
        for _, row in csv_label.iterrows():
            assert (row["left"] < row["right"]), f"coor error found in {self.filename_list[index]}"
            assert (row["top"] < row["bot"]), f"coor error found in {self.filename_list[index]}"
            
            ocr_text.append(row["text"])
            ocr_coor.append([row["left"], row["top"], row["right"], row["bot"]])

        ocr_coor_expand = []
        ocr_tokens = []
        ocr_text_filter = []
        for text, coor in zip(ocr_text, ocr_coor):
            if text == "":
                continue
            curr_tokens = self.tokenizer.tokenize(text)
            for i in range(len(curr_tokens)):
                ocr_coor_expand.append(coor)
                ocr_tokens.append(curr_tokens[i])
                if self.train == False:
                    ocr_text_filter.append(text)

        ocr_corpus = self.tokenizer.convert_tokens_to_ids(ocr_tokens)

        if self.train == True:
            return (
                self.transform_img(image),
                torch.tensor(data_class),
                torch.tensor(pos_neg),
                torch.tensor(ocr_coor_expand, dtype=torch.long),
                torch.tensor(ocr_corpus, dtype=torch.long),
            )
        else:
            return (
                self.transform_img(image),
                torch.tensor(data_class),
                torch.tensor(pos_neg),
                torch.tensor(ocr_coor_expand, dtype=torch.long),
                torch.tensor(ocr_corpus, dtype=torch.long),
                ocr_text_filter,
            )

    def _ViBERTgrid_coll_func(self, samples):
        imgs = []
        class_labels = []
        pos_neg_labels = []
        ocr_coors = []
        ocr_corpus = []
        ocr_text = []
        for item in samples:
            imgs.append(item[0])
            class_labels.append(item[1])
            pos_neg_labels.append(item[2])
            ocr_coors.append(item[3])
            ocr_corpus.append(item[4])
            if self.train == False:
                ocr_text.append(item[5])

        # pad sequence to generate mini-batch
        ocr_coors = pad_sequence(ocr_coors, batch_first=True)
        ocr_corpus = pad_sequence(ocr_corpus, batch_first=True)
        # add mask to indicate valid corpus
        mask = torch.zeros(ocr_corpus.shape, dtype=torch.long)
        mask = mask.masked_fill_((ocr_corpus != 0), 1)

        if self.train == True:
            return (
                tuple(imgs),
                tuple(class_labels),
                tuple(pos_neg_labels),
                ocr_coors.int(),
                ocr_corpus,
                mask.int(),
            )
        else:
            return (
                tuple(imgs),
                tuple(class_labels),
                tuple(pos_neg_labels),
                ocr_coors.int(),
                ocr_corpus,
                mask.int(),
                tuple(ocr_text),
            )


def load_train_dataset(
    root: str,
    batch_size: int,
    num_workers: int = 0,
    tokenizer: Optional[Callable] = None,
    return_mean_std: bool = False,
) -> Tuple[DataLoader]:
    """load EPHOIE train dataset

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
    return_mean_std : bool
        if True, return mean and std of train set, by default False

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    image_mean : numpy.ndarray
    image_std : numpy.ndarray

    """

    EPHOIE_train_dataset = EPHOIEDataset(root=root, train=True, tokenizer=tokenizer)
    EPHOIE_val_dataset = EPHOIEDataset(root=root, train=False, tokenizer=tokenizer)

    train_loader = DataLoader(
        EPHOIE_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=EPHOIE_train_dataset._ViBERTgrid_coll_func,
    )

    val_loader = DataLoader(
        EPHOIE_val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=EPHOIE_val_dataset._ViBERTgrid_coll_func,
    )

    if return_mean_std:
        print("calculating mean and std")
        image_mean = torch.zeros(3)
        image_std = torch.zeros(3)
        for image_list, _, _, _, _, _ in tqdm(train_loader):
            for batch_index in range(batch_size):
                if batch_index >= len(image_list):
                    continue
                curr_img = image_list[batch_index]
                for d in range(3):
                    image_mean[d] += curr_img[d, :, :].mean()
                    image_std[d] += curr_img[d, :, :].std()
        image_mean.div_(len(EPHOIE_train_dataset))
        image_std.div_(len(EPHOIE_train_dataset))

        return train_loader, val_loader, image_mean.numpy(), image_std.numpy()

    return train_loader, val_loader


def load_train_dataset_multi_gpu(
    root: str,
    batch_size: int,
    num_workers: int = 0,
    tokenizer: Optional[Callable] = None,
) -> Tuple[DataLoader]:
    """load EPHOIE train dataset in multi-gpu scene

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
    
    EPHOIE_train_dataset = EPHOIEDataset(root, train=True, tokenizer=tokenizer)
    EPHOIE_val_dataset = EPHOIEDataset(root, train=False, tokenizer=tokenizer)

    train_sampler = distributed.DistributedSampler(EPHOIE_train_dataset)
    val_sampler = distributed.DistributedSampler(EPHOIE_val_dataset)

    train_batch_sampler = BatchSampler(
        train_sampler, batch_size=batch_size, drop_last=True
    )

    train_loader = DataLoader(
        EPHOIE_train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=num_workers,
        collate_fn=EPHOIE_train_dataset._ViBERTgrid_coll_func,
    )

    val_loader = DataLoader(
        EPHOIE_val_dataset,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=EPHOIE_val_dataset._ViBERTgrid_coll_func,
    )

    return train_loader, val_loader, train_sampler


def load_test_data(
    root: str, num_workers: int = 0, tokenizer: Optional[Callable] = None,
):
    EPHOIE_test_dataset = EPHOIEDataset(root=root, train=False, tokenizer=tokenizer)
    test_loader = DataLoader(
        EPHOIE_test_dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=EPHOIE_test_dataset._ViBERTgrid_coll_func,
        shuffle=True,
    )

    return test_loader


if __name__ == "__main__":
    dir_processed = r"dir_to_root"
    model_version = "bert-base-chinese"
    print("loading bert pretrained")
    tokenizer = BertTokenizer.from_pretrained(model_version)
    # train_loader, val_loader, image_mean, image_std = load_train_dataset(
    train_loader, val_loader = load_train_dataset(
        dir_processed,
        batch_size=4,
        num_workers=0,
        tokenizer=tokenizer,
        return_mean_std=False,
    )

    # print(image_mean, image_std)

    for train_batch in tqdm(train_loader):
        img, class_label, pos_neg, coor, corpus, mask = train_batch
        print(class_label[0].shape)
        print(pos_neg[0].shape)
        print(coor.shape)
        print(corpus.shape)
        print(mask.shape)
