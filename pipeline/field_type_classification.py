import torch
import torch.nn as nn

from model.ROI_embedding import ROIEmbedding
from pipeline.roi_align import grid_roi_align


def late_fusion(
    ROI_output: torch.Tensor,
    BERT_embeddings: torch.Tensor,
) -> torch.Tensor:
    bs, seq_len, BERT_dimension = BERT_embeddings.shape

    ROI_embedding_net = ROIEmbedding(
        num_channels=ROI_output.shape[1],
        roi_shape=(ROI_output.shape[2], ROI_output.shape[3])
    )
    # (bs*seq_len, C, ROI_H, ROI_W) -> (bs*seq_len, 1024)
    ROI_embeddings: torch.Tensor = ROI_embedding_net(ROI_output)
    # (bs*seq_len, 1024) + (bs, seq_len, BERT_dimension) -> (bs*seq_len)
    fuse_embeddings = torch.cat(
        (ROI_embeddings, BERT_embeddings.reshape(-1, BERT_dimension)), dim=-1)
    # TODO: activation or not?
    fuse_embedding_net = nn.Sequential([
        nn.Linear(fuse_embeddings.shape[-1], 1024, bias=True)
    ])
    # (bs*seq_len, 1024)
    fuse_embeddings = fuse_embedding_net(fuse_embeddings)

    return fuse_embeddings


# def field_type_classification(
#     fuse_embeddings: torch.Tensor,
#     coords: torch.Tensor,
#     mask: torch.Tensor,
#     pos_neg_label: torch.Tensor,
#     class_label: torch.Tensor
# ):
#     bs = coords.shape[0]
#     field_types = class_label.shape[1] / 2 - 1
#     fuse_embeddings = fuse_embeddings.reshape(bs, -1, fuse_embeddings.shape[-1])
    