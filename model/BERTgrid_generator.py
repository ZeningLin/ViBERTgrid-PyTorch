import torch
import torch.nn as nn

from typing import Tuple, Callable


class BERTgridGenerator(nn.Module):
    """generate BERTgrid with the given OCR results

    This progress can be divided into two sub-progresses,
    BERT embedding and BERTgrid embedding.

    BERT embedding takes corpus and masks from the SROIE dataset
    as input and generate BERT embeddings of the corpus as output.

    BERTgrid embedding maps the BERT embeddings to the feature-maps

    Parameters
    ----------
    bert_model : Callable, optional
        bert_model in hugging face, by default None
    grid_mode : str, optional
        BERTgrid tokens embedding mode.
        words from the OCR result were splited into several tokens through tokenizer,
        at BERTgrid embedding step, measures shall be taken to
        aggregate these token embeddings back into word-level,
        'mean' mode average token embeddings from the same word,
        'first' mode take the first token embeddings of a word,
        by default 'mean'
    stride : int, optional
        stride of the feature map at early fusion, by default 8
    device: str, optional
        device of torch.Tensors, by default 'cpu'

    """

    def __init__(
        self,
        bert_model: Callable = None,
        grid_mode: str = "mean",
        stride: int = 8,
    ) -> None:
        super().__init__()

        assert bert_model is not None, "no bert model given"
        assert grid_mode in [
            "mean",
            "first",
        ], f"grid_mode should be 'mean' or 'first', {grid_mode} were given"

        self.model = bert_model
        self.grid_mode = grid_mode
        self.stride = stride

    def BERT_embedding(
        self, corpus: torch.Tensor, mask: torch.Tensor, seg_indices: torch.Tensor
    ) -> torch.Tensor:
        """apply BERT embedding to the given corpus

        corpus and mask directly come from the SROIEDataset through DataLoader,
        padding and sliding windows will be applied automatically in this function

        Parameters
        ----------
        corpus : torch.Tensor
            corpus from SROIEDataset
        mask : torch.Tensor
            mask from SROIEDataset, indicates whether the token is valid or not
        seg_indices: torch.Tensor
            segment indices from SROIEDataset

        Returns
        -------
        BERT_embeddings : torch.Tensor
            BERT embeddings of the corpus, same in shape with corpus

        """
        device = corpus.device
        self.model = self.model.to(device)

        # if length of sequence exceeds 510 (BERT limitation), apply sliding windows
        batch_size = corpus.shape[0]
        seq_len = corpus.shape[1]
        win_count = seq_len // 510 + 1
        start_index = 0

        # in huggingface, 101 -> [CLS], 102 -> [SEP], 0 -> [PAD]
        cls_token = torch.tensor(
            batch_size * [101], dtype=torch.long, device=device
        ).view(-1, 1)
        sep_token = torch.tensor(
            batch_size * [102], dtype=torch.long, device=device
        ).view(-1, 1)
        # mask tokens for CLS and SEP
        mask_valid = torch.tensor(
            batch_size * [1], dtype=torch.long, device=device
        ).view(-1, 1)

        # apply BERT embedding to all windows
        embeddings = []
        for count in range(win_count):
            end_index = (count + 1) * 510
            curr_seq_len = 0

            # add [CLS] [SEP] [PAD] tokens
            if end_index > seq_len:
                # seq_len < 510, pad
                curr_seq = corpus[:, start_index:]
                curr_seq_len = curr_seq.shape[1]
                curr_mask = mask[:, start_index:]
                pad_tokens = torch.zeros(
                    (corpus.shape[0], (end_index - seq_len)), device=device
                )
                # corpus: [['CLS'] + curr_sequence + ['SEP'] + ['PAD'] * pad_size]
                curr_seq = torch.cat(
                    [cls_token, curr_seq, sep_token, pad_tokens], dim=1
                )
                # mask: [[1] + curr_sequence + [1] + [0] * pad_size]
                curr_mask = torch.cat(
                    [mask_valid, curr_mask, mask_valid, pad_tokens], dim=1
                )
            else:
                curr_seq = corpus[:, start_index:end_index]
                curr_mask = mask[:, start_index:end_index]
                curr_seq_len = curr_seq.shape[1]
                # corpus: [['CLS'] + curr_sequence + ['SEP']]
                curr_seq = torch.cat([cls_token, curr_seq, sep_token], dim=1)
                # mask: [[1] + curr_sequence + [1]]
                curr_mask = torch.cat([mask_valid, curr_mask, mask_valid], dim=1)

            curr_seq = curr_seq.long()
            curr_mask = curr_mask.long()
            # BERT embedding
            curr_output = self.model(input_ids=curr_seq, attention_mask=curr_mask)
            curr_output = curr_output.last_hidden_state

            curr_output = curr_output[
                :,
                1 : (1 + curr_seq_len),
            ]

            embeddings.append(curr_output)

            start_index = end_index

        embeddings = torch.cat(embeddings, dim=1)

        aggre_embeddings = list()
        for batch_index in range(batch_size):
            curr_seg_indices = seg_indices[batch_index]
            curr_embeddings = embeddings[batch_index][mask[batch_index] == 1]
            assert curr_embeddings.shape[0] == curr_seg_indices.shape[0]

            curr_batch_aggre_embeddings = list()

            if self.grid_mode == "mean":
                mean_embeddings = torch.zeros(curr_embeddings.shape[-1], device=device)
                num_tok = 1

            prev_seg_index = -1
            for token_index in range(curr_embeddings.shape[0]):
                curr_seg_index = curr_seg_indices[token_index]
                curr_embedding = curr_embeddings[token_index]
                if curr_seg_index.int().item() == prev_seg_index:
                    if self.grid_mode == "mean":
                        mean_embeddings += curr_embedding
                        num_tok += 1
                    elif self.grid_mode == "first":
                        continue
                else:
                    if self.grid_mode == "mean":
                        mean_embeddings /= num_tok
                        curr_batch_aggre_embeddings.append(mean_embeddings.unsqueeze(0))

                        mean_embeddings = torch.zeros(
                            curr_embeddings.shape[-1], device=device
                        )
                        num_tok = 1
                    elif self.grid_mode == "first":
                        curr_batch_aggre_embeddings.append(curr_embedding.unsqueeze(0))

                prev_seg_index = curr_seg_index.int().item()

            curr_batch_aggre_embeddings = torch.cat(curr_batch_aggre_embeddings, dim=0)
            aggre_embeddings.append(curr_batch_aggre_embeddings)

        return tuple(aggre_embeddings)

    def BERTgrid_embedding(
        self,
        image_shape: Tuple,
        BERT_embeddings: Tuple[torch.Tensor],
        coors: Tuple[torch.Tensor],
    ):
        """map the given BERT embeddings to the BERTgrid

        Parameters
        ----------
        image_shape : Tuple
            shape of the original image
        BERT_embeddings : Tuple[torch.Tensor]
            BERT embeddings from the BERT_embedding function
        coors : Tuple[torch.Tensor]
            coordinate tensor from the SROIEDataset


        Returns
        -------
        BERTgrid : torch.Tensor
            BERTgrid embeddings
        """
        bs = len(BERT_embeddings)
        num_dim = BERT_embeddings[0].shape[-1]
        device = BERT_embeddings[0].device

        BERTgrid = torch.zeros(
            (
                bs,
                num_dim,
                int(image_shape[0] / self.stride),
                int(image_shape[1] / self.stride),
            ),
            device=device,
        )

        for batch_index in range(bs):
            curr_BERT_embeddings = BERT_embeddings[batch_index]
            curr_coors = coors[batch_index]
            assert curr_BERT_embeddings.shape[0] == curr_coors.shape[0]

            for seg_index in range(curr_coors.shape[0]):
                curr_BERT_embedding = curr_BERT_embeddings[seg_index]
                curr_coor = curr_coors[seg_index]
                BERTgrid[
                    batch_index,
                    :,
                    int(curr_coor[1] / self.stride) : int(curr_coor[3] / self.stride),
                    int(curr_coor[0] / self.stride) : int(curr_coor[2] / self.stride),
                ] = curr_BERT_embedding[None, :, None, None]

        return BERTgrid

    def forward(
        self,
        image_shape: Tuple,
        seg_indices: Tuple[torch.Tensor],
        corpus: torch.Tensor,
        mask: torch.Tensor,
        coor: Tuple[torch.Tensor],
    ) -> Tuple[torch.Tensor]:
        """forward propagation

        Parameters
        ----------
        image_shape : Tuple
            shape of the original image
        seg_indices : Tuple[torch.Tensor]
            segment indices from the SROIE dataset
        corpus : torch.Tensor
            corpus from the SROIE dataset
        mask : torch.Tensor
            mask from the SROIE dataset
        coor : torch.Tensor
            coor from the SROIE dataset

        Returns
        -------
        BERT_embeddings: torch.Tensor
            BERT embeddings generated by bert-model

        BERTgrid : torch.Tensor
            BERTgrid generated from OCR results and BERT embeddings
        """
        BERT_embeddings = self.BERT_embedding(
            corpus=corpus, mask=mask, seg_indices=seg_indices
        )
        BERTgrid_embeddings = self.BERTgrid_embedding(
            image_shape=image_shape,
            BERT_embeddings=BERT_embeddings,
            coors=coor,
        )

        return BERT_embeddings, BERTgrid_embeddings


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    from data.SROIE_dataset import load_train_dataset
    from utils.ViBERTgrid_visualize import ViBERTgrid_visualize
    from pipeline.transform import GeneralizedViBERTgridTransform
    from transformers import BertTokenizer, BertModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    args = parser.parse_args()

    image_mean = [0.9248, 0.9224, 0.9215]
    image_std = [0.1532, 0.1545, 0.1536]
    image_min_size = [320, 416, 512, 608, 704]
    test_image_min_size = 500
    image_max_size = 800

    dir_processed = args.root
    model_version = "./private_bert-base-uncased"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("loading bert pretrained")
    tokenizer = BertTokenizer.from_pretrained(model_version)
    model = BertModel.from_pretrained(model_version)

    train_loader, val_loader = load_train_dataset(
        dir_processed, batch_size=2, num_workers=0, tokenizer=tokenizer
    )
    transform = GeneralizedViBERTgridTransform(
        image_mean=image_mean,
        image_std=image_std,
        train_min_size=image_min_size,
        test_min_size=test_image_min_size,
        max_size=image_max_size,
    )

    img, seg_indices, token_classes, coor, corpus, mask = next(iter(train_loader))
    ViBERTgrid_generator = BERTgridGenerator(
        bert_model=model, grid_mode="mean", stride=8
    )

    ViBERTgrid_generator = ViBERTgrid_generator.to(device)

    img_convert = list(img)

    img, coor = transform(img, coor)

    BERT_embeddings, BERTgrid_embeddings = ViBERTgrid_generator(
        img.tensors.shape[-2:], seg_indices, corpus, mask, coor
    )

    num_pic = BERTgrid_embeddings.shape[0]
    img_convert = [img.permute(1, 2, 0).detach().numpy() for img in img_convert]
    grid_convert = torch.mean(BERTgrid_embeddings.float(), dim=1).detach().numpy()
    grid_convert *= 255
    for i in range(num_pic):
        curr_grid = grid_convert[i]
        curr_img = img_convert[i]
        plt.imsave(f"./temp/orig_img{i}.jpg", curr_img)
        plt.imsave(f"./temp/BERTgrid_{i}.jpg", curr_grid)
