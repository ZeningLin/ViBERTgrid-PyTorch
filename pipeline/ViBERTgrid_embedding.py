import torch
from transformers import BertModel

from typing import Tuple, Optional, Callable


def BERT_embedding(
    corpus: torch.Tensor,
    mask: torch.Tensor,
    bert_model: Callable = None,
    mode='train'
) -> torch.tensor:
    """apply BERT embedding to the given corpus 
     
    corpus and mask directly come from the SROIEDataset through DataLoader,   
    padding and sliding windows will be applied automatically in this function

    Parameters
    ----------
    corpus : torch.Tensor
        corpus from SROIEDataset
    mask : torch.Tensor
        mask from SROIEDataset, indicates whether the token is valid or not
    bert_model : Callable, optional
        bert_model in hugging face, by default None
    mode : str, optional
        'train' or 'eval' to control back-propagation, by default 'train'

    Returns
    -------
    BERT_embeddings : torch.tensor
        BERT embeddings of the corpus, same in shape with corpus
    """

    assert bert_model is not None, 'no bert model given'
    assert mode in [
        'train', 'eval'], f"invalid mode '{mode}', must be 'train' or 'eval'"
    model = bert_model

    if(mode == 'train'):
        model.train()
    else:
        model.eval()

    # if length of sequence exceeds 510 (BERT limitation), apply sliding windows
    batch_size = corpus.shape[0]
    seq_len = corpus.shape[1]
    win_count = seq_len // 510 + 1
    start_index = 0

    # in huggingface, 101 -> [CLS], 102 -> [SEP], 0 -> [PAD]
    cls_token = torch.tensor(batch_size * [101], dtype=torch.long).view(-1, 1)
    sep_token = torch.tensor(batch_size * [102], dtype=torch.long).view(-1, 1)
    # mask tokens for CLS and SEP
    mask_valid = torch.tensor(batch_size * [1], dtype=torch.long).view(-1, 1)

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
            pad_tokens = torch.zeros((corpus.shape[0], (end_index - seq_len)))
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
        curr_output = model(input_ids=curr_seq, attention_mask=curr_mask)
        curr_output = curr_output.last_hidden_state

        curr_output = curr_output[:, 1:(1+curr_seq_len), ]

        embeddings.append(curr_output)

        start_index = end_index

    if(win_count == 1):
        return embeddings[0]
    else:
        return torch.cat(embeddings, dim=1)


def ViBERTgrid_embedding(
    BERT_embeddings: torch.Tensor,
    coors: torch.Tensor,
    mask: torch.Tensor,
    img_shape: Tuple,
    stride: int = 8,
    mode: str = 'mean'
) -> torch.Tensor:
    """map the given BERT embeddings to the ViBERTgrid

    Parameters
    ----------
    BERT_embeddings : torch.Tensor
        BERT embeddings from the BERT_embedding function
    coors : torch.Tensor
        coordinate tensor from the SROIEDataset
    mask : torch.Tensor
        mask tensor from the SROIEDataset
    img_shape : Tuple
        shape of input features of ResNet18
    stride : int, optional
        stride of the feature map at early fusion, by default 8
    mode : str, optional
        ViBERTgrid tokens embedding mode.
        words from the OCR result were splited into several tokens through tokenizer,  
        at ViBERTgrid embedding step, measures shall be taken to  
        aggregate these token embeddings back into word-level,  
        'mean' mode average token embeddings from the same word,  
        'first' mode take the first token embeddings of a word,  
        by default 'mean'

    Returns
    -------
    ViBERTgrid : torch.Tensor
        ViBERTgrid embeddings
    """
    
    assert mode in [
        'mean', 'first'], f"mode should be 'mean' or 'first', {mode} were given"

    bs, seq_len, num_dim = BERT_embeddings.shape

    ViBERTgrid = torch.zeros(
        (bs, num_dim, int(img_shape[0] / stride), int(img_shape[1] / stride)))

    for bs_index in range(bs):
        # TODO  low efficiency implementation, may optimized later
        #       1. unable to vectorize due to the corpus length difference inside a batch
        
        # initialize coors with [-1, -1, -1, -1] to match first coor
        prev_coors = torch.ones((coors.shape[2]), dtype=torch.long) * (-1)
        if mode == 'mean':
            mean_count = 1
            curr_embs = torch.zeros((num_dim), dtype=torch.float32)
        for seq_index in range(seq_len):
            if mask[bs_index, seq_index] == 0:
                break
            if coors[bs_index, seq_index, :].equal(prev_coors):
                if mode == 'mean':
                    curr_embs += BERT_embeddings[bs_index, seq_index, :]
                    mean_count += 1
                elif mode == 'first':
                    continue
            else:
                word_coors = coors[bs_index, seq_index] / stride
                word_coors = word_coors.int()

                if mode == 'mean':
                    ViBERTgrid[bs_index, :, word_coors[1]: word_coors[3], word_coors[0]: word_coors[2]] = \
                        (curr_embs[:, None, None] / mean_count)
                    mean_count = 1
                    curr_embs = BERT_embeddings[bs_index, seq_index, :]
                elif mode == 'first':
                    ViBERTgrid[bs_index, :, word_coors[1]: word_coors[3], word_coors[0]: word_coors[2]] = \
                        BERT_embeddings[bs_index, seq_index, :, None, None]

            prev_coors = coors[bs_index, seq_index, :]

    return ViBERTgrid


if __name__ == '__main__':
    import sys
    sys.path.append('..')

    from data.SROIE_dataset import load_train_dataset
    from utils.ViBERTgrid_visualize import ViBERTgrid_visualize
    from transformers import BertTokenizer

    RESIZE_SHAPE = (336, 256)

    dir_processed = r'D:\PostGraduate\DataSet\ICDAR-SROIE\ViBERTgrid_format\train'
    model_version = 'bert-base-uncased'
    device = 'cpu'
    print('loading bert pretrained')
    tokenizer = BertTokenizer.from_pretrained(model_version)
    model = BertModel.from_pretrained(model_version)
    model = model.to(device)

    train_loader, val_loader = load_train_dataset(
        dir_processed,
        batch_size=4,
        val_ratio=0.3,
        num_workers=0,
        tokenizer=tokenizer
    )

    for img, class_label, pos_neg_label, coor, corpus, mask in train_loader:
        BERTemb = BERT_embedding(corpus, mask, model)
        ViBERTgrid = ViBERTgrid_embedding(
            BERTemb, coor, mask, RESIZE_SHAPE, stride=8, mode='mean')
        print(ViBERTgrid.shape)
        ViBERTgrid_visualize(ViBERTgrid)
