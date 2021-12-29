import torch


@torch.no_grad()
def SROIE_label_classification_criteria(
    gt_label: torch.Tensor, pred_label: torch.Tensor
):
    num_correct = 0.0
    num_entities = gt_label.shape[0]
    for entity_index in range(num_entities):
        if gt_label[entity_index] == pred_label[entity_index]:
            num_correct += 1

    return num_correct, num_entities


@torch.no_grad()
def SROIE_label_F1_criteria(gt_label: torch.Tensor, pred_label: torch.Tensor):
    num_entities = gt_label.shape[0]

    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0

    # TODO definition of TP/TN/FP/FN ?
    for entity_index in range(num_entities):
        if gt_label[entity_index] == pred_label[entity_index]:
            if gt_label[entity_index] == 0:
                TN += 1
            else:
                TP += 1
        else:
            if pred_label[entity_index] != 0:
                FP +=1
            else:
                FN += 1

    return TP, TN, FP, FN


@torch.no_grad()
def SROIE_ss_classification_criteria(
    pred_ss_label: torch.Tensor, class_ss_label: torch.Tensor, coor: torch.Tensor
):
    batch_size = pred_ss_label.shape[0]
    num_entities = coor.shape[2]
    classify_correct = 0.0
    for batch_index in range(batch_size):
        for entity_index in range(num_entities):
            curr_coor = coor[batch_index, entity_index]
            gt_label = class_ss_label[
                batch_index, :, curr_coor[1] : curr_coor[3], curr_coor[0] : curr_coor[2]
            ].argmax(dim=0)
            pred_label = pred_ss_label[
                batch_index, :, curr_coor[1] : curr_coor[3], curr_coor[0] : curr_coor[2]
            ].argmax(dim=0)
            if gt_label == pred_label:
                classify_correct += 1

    return classify_correct, num_entities
