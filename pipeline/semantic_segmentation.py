import torch
import torch.nn as nn


from model.semantic_classification_net import SemanticSegmentationNet


def semantic_segmentation(
    fuse_feature: torch.Tensor,
    pos_neg_labels: torch.Tensor,
    class_labels: torch.Tensor
) -> torch.Tensor:
    """auxiliary semantic segmentation head,  
       apply two multi-class classification to the feature map,

    Parameters
    ----------
    fuse_feature : torch.Tensor
        p_fuse feature maps mentioned in sec 3.1.2 of the paper
    pos_neg_labels : torch.Tensor
        pos_neg labels from SROIEDataset
    class_labels : torch.Tensor
        class labels from SROIEDataset

    Returns
    -------
    aux_loss : torch.Tensor
        auxiliary segmentation loss
    """
    device = fuse_feature.device
    fuse_channel = fuse_feature.shape[1]

    aux_loss_1 = nn.CrossEntropyLoss()
    aux_loss_2 = nn.CrossEntropyLoss()
    semantic_segmentation_net = SemanticSegmentationNet(
        fuse_channel=fuse_channel)
    semantic_segmentation_net = semantic_segmentation_net.to(device)
    x_out_1, x_out_2 = semantic_segmentation_net(fuse_feature)

    aux_loss_1_val = aux_loss_1(x_out_1, pos_neg_labels.argmax(dim=1))
    aux_loss_2_val = aux_loss_2(x_out_2, class_labels.argmax(dim=1))

    return aux_loss_1_val + aux_loss_2_val


if __name__ == '__main__':
    from data.SROIE_dataset import load_train_dataset
    from transformers import BertTokenizer
    from tqdm import tqdm

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    total_loss = 0
    num_batch = len(train_loader)
    for train_batch in tqdm(train_loader):
        img, class_label, pos_neg, coor, corpus, mask = train_batch
        class_label = class_label.to(device)
        pos_neg = pos_neg.to(device)
        coor = coor.to(device)
        corpus = corpus.to(device)
        mask = mask.to(device)
        fuse = torch.zeros(class_label.shape[0], 256, int(
            class_label.shape[2] / 4), int(class_label.shape[3] / 4), device=device)
        loss = semantic_segmentation(
            fuse_feature=fuse,
            pos_neg_labels=pos_neg,
            class_labels=class_label
        )
        loss.backward()

        total_loss += loss

    print(total_loss)
