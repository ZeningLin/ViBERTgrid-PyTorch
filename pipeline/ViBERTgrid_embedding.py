import torch
from transformers import BertModel

from typing import Tuple, Optional, Callable


def BERT_embedding(
    corpus: torch.Tensor,
    mask: torch.Tensor,
    bert_model: Callable = None,
    mode='train'
) -> torch.tensor:

    assert bert_model is not None, 'no bert model given'
    assert mode in [
        'train', 'eval'], 'invalid mode \'{}\', must be \'train\' or \'eval\''.format(mode)
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
    img_shape: Tuple,
    stride: int = 8
) -> torch.Tensor:
    bs, seq_len, num_dim = BERT_embeddings.shape
    ViBERTgrid = torch.zeros(
        (bs, num_dim, int(img_shape[0] / stride), int(img_shape[1] / stride)))
    for seq_index in range(seq_len):
        curr_coors = coors[:, seq_index, :] / stride
        curr_coors = curr_coors.int()
        curr_emb = BERT_embeddings[:, seq_index, :]
        for bs_index in range(bs):
            ViBERTgrid[bs_index, :,
                       curr_coors[bs_index, 1]: curr_coors[bs_index, 3],
                       curr_coors[bs_index, 0]: curr_coors[bs_index, 2]] = \
                curr_emb[bs_index, :, None,
                         None]  # expand dimensions to match requirements

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
            BERTemb, coor, RESIZE_SHAPE, stride=8)
        print(ViBERTgrid.shape)
        ViBERTgrid_visualize(ViBERTgrid)
