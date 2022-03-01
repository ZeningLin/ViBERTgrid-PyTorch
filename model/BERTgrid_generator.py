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
        grid_mode: str = 'mean',
        stride: int = 8,
    ) -> None:
        super().__init__()

        assert bert_model is not None, 'no bert model given'
        assert grid_mode in ['mean', 'first'], f"grid_mode should be 'mean' or 'first', {grid_mode} were given"

        self.model = bert_model
        self.grid_mode = grid_mode
        self.stride = stride
        
    def BERT_embedding(
        self,
        corpus: torch.Tensor,
        mask: torch.Tensor
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
            batch_size * [101], dtype=torch.long, device=device).view(-1, 1)
        sep_token = torch.tensor(
            batch_size * [102], dtype=torch.long, device=device).view(-1, 1)
        # mask tokens for CLS and SEP
        mask_valid = torch.tensor(
            batch_size * [1], dtype=torch.long, device=device).view(-1, 1)

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
                    (corpus.shape[0], (end_index - seq_len)), device=device)
                # corpus: [['CLS'] + curr_sequence + ['SEP'] + ['PAD'] * pad_size]
                curr_seq = torch.cat(
                    [cls_token, curr_seq, sep_token, pad_tokens], dim=1)
                # mask: [[1] + curr_sequence + [1] + [0] * pad_size]
                curr_mask = torch.cat(
                    [mask_valid, curr_mask, mask_valid, pad_tokens], dim=1)
            else:
                curr_seq = corpus[:, start_index: end_index]
                curr_mask = mask[:, start_index: end_index]
                curr_seq_len = curr_seq.shape[1]
                # corpus: [['CLS'] + curr_sequence + ['SEP']]
                curr_seq = torch.cat([cls_token, curr_seq, sep_token], dim=1)
                # mask: [[1] + curr_sequence + [1]]
                curr_mask = torch.cat(
                    [mask_valid, curr_mask, mask_valid], dim=1)

            curr_seq = curr_seq.long()
            curr_mask = curr_mask.long()
            # BERT embedding
            curr_output = self.model(
                input_ids=curr_seq, attention_mask=curr_mask)
            curr_output = curr_output.last_hidden_state

            curr_output = curr_output[:, 1:(1+curr_seq_len), ]

            embeddings.append(curr_output)

            start_index = end_index

        if(win_count == 1):
            return embeddings[0]
        else:
            return torch.cat(embeddings, dim=1)
    
    def BERTgrid_embedding(
        self,
        image_shape: Tuple,
        BERT_embeddings: torch.Tensor,
        coors: torch.Tensor,
        mask: torch.Tensor,
    ):
        """map the given BERT embeddings to the BERTgrid

        Parameters
        ----------
        image_shape : Tuple
            shape of the original image
        BERT_embeddings : torch.Tensor
            BERT embeddings from the BERT_embedding function
        coors : torch.Tensor
            coordinate tensor from the SROIEDataset
        mask : torch.Tensor
            mask tensor from the SROIEDataset
        
        Returns
        -------
        BERTgrid : torch.Tensor
            BERTgrid embeddings
        """
        bs, seq_len, num_dim = BERT_embeddings.shape
        device = BERT_embeddings.device

        BERTgrid = torch.zeros(
            (bs, num_dim, int(image_shape[0] / self.stride), int(image_shape[1] / self.stride)), device=device)

        for bs_index in range(bs):
            # TODO  low efficiency implementation, may optimized later
            #       1. unable to vectorize due to the corpus length difference inside a batch

            # initialize coors with [-1, -1, -1, -1] to match first coor
            prev_coors = torch.ones(
                (coors.shape[2]), dtype=torch.int, device=device)
            prev_coors *= -1
            if self.grid_mode == 'mean':
                mean_count = 1
                curr_embs = torch.zeros(
                    (num_dim), dtype=torch.float32, device=device)
            for seq_index in range(seq_len):
                if mask[bs_index, seq_index] == 0:
                    break
                if coors[bs_index, seq_index, :].equal(prev_coors):
                    if self.grid_mode == 'mean':
                        curr_embs += BERT_embeddings[bs_index, seq_index, :]
                        mean_count += 1
                    elif self.grid_mode == 'first':
                        continue
                else:
                    word_coors = coors[bs_index, seq_index] / self.stride
                    word_coors = word_coors.int()

                    if self.grid_mode == 'mean':
                        BERTgrid[bs_index, :, word_coors[1]: word_coors[3], word_coors[0]: word_coors[2]] = \
                            (curr_embs[:, None, None] / mean_count)
                        mean_count = 1
                        curr_embs = BERT_embeddings[bs_index, seq_index, :]
                    elif self.grid_mode == 'first':
                        BERTgrid[bs_index, :, word_coors[1]: word_coors[3], word_coors[0]: word_coors[2]] = \
                            BERT_embeddings[bs_index, seq_index, :, None, None]

                prev_coors = coors[bs_index, seq_index, :]

        return BERTgrid
    
    def forward(
        self,
        image_shape: Tuple,
        corpus: torch.Tensor,
        mask: torch.Tensor,
        coor: torch.Tensor
    ) -> torch.Tensor:
        """forward propagation

        Parameters
        ----------
        image_shape : Tuple
            shape of the original image
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
            corpus=corpus,
            mask=mask
        )
        BERTgrid_embeddings = self.BERTgrid_embedding(
            image_shape=image_shape, 
            BERT_embeddings=BERT_embeddings,
            coors=coor,
            mask=mask
        )
        
        return BERT_embeddings, BERTgrid_embeddings


if __name__ == '__main__':
    import sys
    
    sys.path.append('..')

    from data.SROIE_dataset import load_train_dataset
    from utils.ViBERTgrid_visualize import ViBERTgrid_visualize
    from transformers import BertTokenizer, BertModel

    RESIZE_SHAPE = (336, 256)

    dir_processed = r'dir_to_data'
    model_version = 'bert-base-uncased'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('loading bert pretrained')
    tokenizer = BertTokenizer.from_pretrained(model_version)
    model = BertModel.from_pretrained(model_version)

    train_loader, val_loader = load_train_dataset(
        dir_processed,
        batch_size=4,
        num_workers=0,
        tokenizer=tokenizer
    )

    for img, class_label, pos_neg_label, coor, corpus, mask in train_loader:
        ViBERTgrid_generator = BERTgridGenerator(
            bert_model=model,
            pipeline_mode='train',
            grid_mode='mean',
            stride=8
        )
        ViBERTgrid_generator = ViBERTgrid_generator.to(device)
        
        BERTgrid = ViBERTgrid_generator(
            corpus,
            mask,
            coor
        )

        print(BERTgrid.shape)
        ViBERTgrid_visualize(BERTgrid)
