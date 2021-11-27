import torch
import torch.nn as nn

from model.field_type_classification_model import ROIEmbedding, SingleLayer
from pipeline.roi_align import grid_roi_align


def late_fusion(
    ROI_output: torch.Tensor,
    BERT_embeddings: torch.Tensor,
) -> torch.Tensor:
    _, _, BERT_dimension = BERT_embeddings.shape

    ROI_embedding_net = ROIEmbedding(
        num_channels=ROI_output.shape[1],
        roi_shape=(ROI_output.shape[2], ROI_output.shape[3])
    )
    # (bs*seq_len, C, ROI_H, ROI_W) -> (bs*seq_len, 1024)
    ROI_embeddings: torch.Tensor = ROI_embedding_net(ROI_output)
    # (bs*seq_len, 1024) + (bs, seq_len, BERT_dimension) -> (bs*seq_len)
    fuse_embeddings = torch.cat(
        (ROI_embeddings, BERT_embeddings.reshape(-1, BERT_dimension)), dim=1)

    fuse_embedding_net = SingleLayer(
        in_channels=fuse_embeddings.shape[-1],
        out_channels=1024,
        bias=True
    )

    # (bs*seq_len, 1024)
    fuse_embeddings = fuse_embedding_net(fuse_embeddings)

    return fuse_embeddings


def field_type_classification(
    fuse_embeddings: torch.Tensor,
    coords: torch.Tensor,
    mask: torch.Tensor,
    class_labels: torch.Tensor
):
    bs = coords.shape[0]
    seq_len = coords.shape[1]
    field_types = class_labels.shape[1]
    # (bs*seq_len, 1024) -> (bs, seq_len, 1024)
    fuse_embeddings = fuse_embeddings.reshape(
        bs, -1, fuse_embeddings.shape[-1])

    classification_net = SingleLayer(
        in_channels=fuse_embeddings.shape[-1],
        out_channels=field_types,
        bias=True
    )

    # (bs*seq_len, field_types)
    pred_class_orig = classification_net(
        fuse_embeddings.reshape(-1, fuse_embeddings.shape[-1]))
    pred_class_orig = pred_class_orig.reshape(bs, seq_len, field_types)
    # TODO Low efficiency implementation, need optimization
    classification_loss = nn.CrossEntropyLoss()
    classification_loss_val = 0
    label_class = []
    pred_class = []
    for bs_index in range(bs):
        for seq_index in range(seq_len):
            if mask[bs_index, seq_index] == 1:
                cur_coor = coords[bs_index, seq_index, :]
                curr_label_class = class_labels[bs_index, :, cur_coor[1]:cur_coor[3],
                                                cur_coor[0]:cur_coor[2]]
                curr_label_class = curr_label_class.argmax(dim=0).reshape(-1)
                curr_label_class = curr_label_class.bincount().argmax().item()
                label_class.append(curr_label_class)
                pred_class.append(pred_class_orig[None, bs_index, seq_index])
            else:
                continue

    label_class = torch.tensor(label_class)
    pred_class = torch.cat(pred_class, dim=0)
    # TODO computing CELoss is time-comsuming
    classification_loss_val = classification_loss(pred_class, label_class)
    return classification_loss_val


if __name__ == '__main__':
    from data.SROIE_dataset import load_train_dataset
    from roi_align import grid_roi_align
    from pipeline.ViBERTgrid_embedding import BERT_embedding, ViBERTgrid_embedding

    from transformers import BertTokenizer, BertModel
    from tqdm import tqdm

    RESIZE_SHAPE = (336, 256)
    dir_processed = r'D:\PostGraduate\DataSet\ICDAR-SROIE\ViBERTgrid_format\train'
    model_version = 'bert-base-uncased'
    print('loading bert pretrained')
    tokenizer = BertTokenizer.from_pretrained(model_version)
    bert_model = BertModel.from_pretrained(model_version)
    train_loader, val_loader = load_train_dataset(
        dir_processed,
        batch_size=4,
        val_ratio=0.3,
        num_workers=0,
        tokenizer=tokenizer
    )

    total_loss = 0
    num_batch = len(train_loader)
    for train_batch in tqdm(train_loader):
        img, class_label, pos_neg, coor, corpus, mask = train_batch
        BERTemb = BERT_embedding(corpus, mask, bert_model)
        p_fuse = torch.zeros(class_label.shape[0], 256, int(
            class_label.shape[2] / 4), int(class_label.shape[3] / 4))
        roi_output = grid_roi_align(
            feature_map=p_fuse,
            coords=coor.float(),
            mask=None,
            output_size=7,
            step=4,
            output_reshape=False
        )
        fuse_embeddings = late_fusion(
            ROI_output=roi_output,
            BERT_embeddings=BERTemb
        )
        classification_loss = field_type_classification(
            fuse_embeddings=fuse_embeddings,
            coords=coor,
            mask=mask,
            class_labels=class_label
        )
        classification_loss.backward()

        print(p_fuse.shape)
        print(roi_output.shape)
        print(fuse_embeddings.shape)
        print(classification_loss)
