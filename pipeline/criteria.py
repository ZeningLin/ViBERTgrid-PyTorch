import torch
from typing import Dict


@torch.no_grad()
def token_classification_criteria(gt_label: torch.Tensor, pred_label: torch.Tensor):
    pred_label = pred_label.argmax(dim=1).int()
    num_correct = 0.0
    num_entities = gt_label.shape[0]
    for entity_index in range(num_entities):
        if gt_label[entity_index] == pred_label[entity_index]:
            num_correct += 1

    return num_correct, num_entities


@torch.no_grad()
def token_F1_criteria(pred_gt_dict: Dict[torch.Tensor, torch.Tensor]):
    pred_label: torch.Tensor
    gt_label: torch.Tensor
    pred_label = torch.cat(list(pred_gt_dict.keys()), dim=0)
    gt_label = torch.cat(list(pred_gt_dict.values()), dim=0)

    num_classes = pred_label.shape[1]
    pred_label = pred_label.argmax.int()

    result_dict = dict()
    for class_index in range(num_classes):
        curr_gt_index = gt_label == class_index
        TP = (pred_label[curr_gt_index, class_index] == 1).int().sum().item()
        TN = (pred_label[~curr_gt_index, class_index] == 0).int().sum().item()
        FP = (pred_label[~curr_gt_index, class_index] == 1).int().sum().item()
        FN = (pred_label[curr_gt_index, class_index] == 0).int().sum().item()

        curr_precision = TP / (TP + FP + 1e-8)
        curr_recall = TP / (TP + FN + 1e-8)
        curr_F1 = (
            2 * curr_precision * curr_recall / (curr_precision + curr_recall + 1e-8)
        )

        curr_class_dict = {
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "precision": curr_precision,
            "recall": curr_recall,
            "F1": curr_F1,
        }
        result_dict.update({class_index: curr_class_dict})

    num_correct = (pred_label == gt_label).int().sum()
    num_total = gt_label.shape[0]
    result_dict.update({"num_classes": num_classes})
    result_dict.update({"num_correct": num_correct})
    result_dict.update({"num_total": num_total})

    return result_dict


@torch.no_grad()
def semantic_segmentation_classification_criteria(
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
