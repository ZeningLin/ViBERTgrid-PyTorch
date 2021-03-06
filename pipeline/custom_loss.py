import torch
import torch.nn as nn
import torch.nn.functional as F

import random
from typing import Optional, List


class CrossEntropyLossRandomSample(nn.CrossEntropyLoss):
    def __init__(
        self,
        sample_list: List,
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        self.sample_list = sample_list
        if sample_list is not None:
            assert (
                len(sample_list) >= 2
            ), f"sample list must contains at least two elements, {len(sample_list)} given"
            self.num_categories = len(sample_list)
        else:
            self.num_categories = None

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sample_list is None:
            return F.cross_entropy(
                input.float(),
                target,
                weight=self.weight,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
                label_smoothing=self.label_smoothing,
            )

        ce_loss = F.cross_entropy(
            input.float(),
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        mask_list = []
        if self.num_categories == 2 and input.shape[1] >= 2:
            mask_list = [(target == 0), (target != 0)]
        else:
            assert (
                self.num_categories == input.shape[1]
            ), f"shape mismatch, number of elements in sample_list must be 2 or equals dimensions \
                of input, {self.num_categories} and {input.shape[1]} given"
            for cate_index in range(self.num_categories):
                mask_list.append(target == cate_index)

        loss_list = []
        num_keep_list = []
        for index, mask in enumerate(mask_list):
            curr_sample = self.sample_list[index]
            curr_loss = ce_loss[mask]
            num_keep = min(curr_sample, curr_loss.shape[0])
            num_keep_list.append(num_keep)

            if num_keep == curr_sample:
                keep_index = random.sample(range(int(curr_loss.shape[0])), num_keep)
                keep_index = torch.tensor(keep_index, device=curr_loss.device)
                keep_loss = curr_loss[keep_index]
            else:
                keep_loss = curr_loss

            loss_list.append(keep_loss)

        if self.reduction == "sum":
            keep_ce_loss = torch.zeros((1,), dtype=float, device=target.device)
            for loss in loss_list:
                keep_ce_loss = keep_ce_loss + loss.sum()
        elif self.reduction == "mean":
            num_keep_total = 0
            keep_ce_loss = torch.zeros((1,), dtype=float, device=target.device)
            for loss, num_keep in zip(loss_list, num_keep_list):
                keep_ce_loss = keep_ce_loss + loss.sum()
                num_keep_total = num_keep_total + num_keep
            keep_ce_loss /= num_keep_total
        elif self.reduction == "none":
            keep_ce_loss = torch.stack(loss_list)
        else:
            raise ValueError(
                f"the given reduction value {self.reduction} is invalid, must be 'none', 'mean' or 'sum' "
            )

        return keep_ce_loss


class CrossEntropyLossOHEM(nn.CrossEntropyLoss):
    def __init__(
        self,
        num_hard_positive: int = -1,
        num_hard_negative: int = -1,
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        random: bool = False,
    ) -> None:
        super().__init__(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        self.num_hard_positive = num_hard_positive
        self.num_hard_negative = num_hard_negative
        self.random = random

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.num_hard_positive == -1 and self.num_hard_negative == -1:
            return F.cross_entropy(
                input.float(),
                target,
                weight=self.weight,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
                label_smoothing=self.label_smoothing,
            )

        ce_loss = F.cross_entropy(
            input.float(),
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        mask = target == 0
        positive_loss = ce_loss[~mask]
        negative_loss = ce_loss[mask]

        if self.random:
            num_random_pos = 2 * self.num_hard_positive
            if num_random_pos < positive_loss.shape[0]:
                keep_index = random.sample(
                    range(int(positive_loss.shape[0])), num_random_pos
                )
                keep_index = torch.tensor(keep_index, device=ce_loss.device)
                positive_loss = positive_loss[keep_index]

            num_random_neg = 2 * self.num_hard_negative
            if num_random_neg < negative_loss.shape[0]:
                keep_index = random.sample(
                    range(int(negative_loss.shape[0])), num_random_neg
                )
                keep_index = torch.tensor(keep_index, device=ce_loss.device)
                negative_loss = negative_loss[keep_index]

        sorted_positive_loss, sorted_positive_index = torch.sort(
            positive_loss, descending=True
        )
        num_positive_keep = min(sorted_positive_loss.shape[0], self.num_hard_positive)
        if num_positive_keep <= 0:
            pass
        elif num_positive_keep < sorted_positive_loss.shape[0]:
            keep_pos_index = sorted_positive_index[:num_positive_keep]
            sorted_positive_loss = sorted_positive_loss[keep_pos_index]

        sorted_negative_loss, sorted_negative_index = torch.sort(
            negative_loss, descending=True
        )
        num_negative_keep = min(sorted_negative_loss.shape[0], self.num_hard_negative)
        if num_negative_keep <= 0:
            pass
        elif num_negative_keep < sorted_negative_loss.shape[0]:
            keep_neg_index = sorted_negative_index[:num_negative_keep]
            sorted_negative_loss = sorted_negative_loss[keep_neg_index]

        if self.reduction == "sum":
            keep_ce_loss = sorted_positive_loss.sum() + sorted_negative_loss.sum()
        elif self.reduction == "mean":
            keep_ce_loss = (sorted_positive_loss.sum() + sorted_negative_loss.sum()) / (
                num_positive_keep + num_negative_keep
            )
        elif self.reduction == "none":
            keep_ce_loss = torch.stack([sorted_positive_loss, sorted_negative_loss])
        else:
            raise ValueError(
                f"the given reduction value {self.reduction} is invalid, must be 'none', 'mean' or 'sum' "
            )

        return keep_ce_loss


class BCELossRandomSample(nn.BCELoss):
    def __init__(
        self,
        sample_list: List,
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(
            weight=weight,
            size_average=size_average,
            reduction=reduction,
        )
        self.sample_list = sample_list
        if sample_list is not None:
            assert (
                len(sample_list) <= 2
            ), f"sample list must contains at most two elements, {len(sample_list)} given"
            if len(sample_list) == 2:
                sample_list = [sample_list[0], sample_list[0]]
            self.num_categories = 2
        else:
            self.num_categories = None

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert len(input.shape) == 1 or (
            len(input.shape) == 2 and input.shape[1] == 1
        ), f"invalid shape"
        if len(input.shape) == 2:
            input = input.squeeze(1)

        if self.sample_list is None:
            return F.binary_cross_entropy_with_logits(
                input.float(),
                target,
                weight=self.weight,
                reduction=self.reduction,
            )

        ce_loss = F.binary_cross_entropy_with_logits(
            input.float(),
            target,
            weight=self.weight,
            reduction="none",
        )

        mask = input > 0

        loss_list = []
        num_keep_list = []
        for index in range(2):
            curr_sample = self.sample_list[index]
            if index == 0:
                curr_loss = ce_loss[~mask]
            else:
                curr_loss = ce_loss[mask]
            num_keep = min(curr_sample, curr_loss.shape[0])
            num_keep_list.append(num_keep)

            if num_keep == curr_sample:
                keep_index = random.sample(range(int(curr_loss.shape[0])), num_keep)
                keep_index = torch.tensor(keep_index, device=curr_loss.device)
                keep_loss = curr_loss[keep_index]
            else:
                keep_loss = curr_loss

            loss_list.append(keep_loss)

        if self.reduction == "sum":
            keep_ce_loss = torch.zeros((1,), dtype=float, device=target.device)
            for loss in loss_list:
                keep_ce_loss = keep_ce_loss + loss.sum()
        elif self.reduction == "mean":
            num_keep_total = 0
            keep_ce_loss = torch.zeros((1,), dtype=float, device=target.device)
            for loss, num_keep in zip(loss_list, num_keep_list):
                keep_ce_loss = keep_ce_loss + loss.sum()
                num_keep_total = num_keep_total + num_keep
            keep_ce_loss /= num_keep_total
        elif self.reduction == "none":
            keep_ce_loss = torch.stack(loss_list)
        else:
            raise ValueError(
                f"the given reduction value {self.reduction} is invalid, must be 'none', 'mean' or 'sum' "
            )

        return keep_ce_loss


class BCELossOHEM(nn.BCELoss):
    def __init__(
        self,
        num_hard_positive: int = -1,
        num_hard_negative: int = -1,
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        reduction: str = "mean",
        random: bool = False,
    ) -> None:
        super().__init__(
            weight=weight,
            size_average=size_average,
            reduction=reduction,
        )
        self.num_hard_positive = num_hard_positive
        self.num_hard_negative = num_hard_negative
        self.random = random

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.num_hard_positive == -1 and self.num_hard_negative == -1:
            return F.binary_cross_entropy_with_logits(
                input.float(),
                target,
                weight=self.weight,
                reduction=self.reduction,
            )

        ce_loss = F.binary_cross_entropy_with_logits(
            input.float(),
            target,
            weight=self.weight,
            reduction="none",
        )

        mask = target == 0
        positive_loss = ce_loss[~mask]
        negative_loss = ce_loss[mask]

        if self.random:
            num_random_pos = 2 * self.num_hard_positive
            if num_random_pos < positive_loss.shape[0]:
                keep_index = random.sample(
                    range(int(positive_loss.shape[0])), num_random_pos
                )
                keep_index = torch.tensor(keep_index, device=ce_loss.device)
                positive_loss = positive_loss[keep_index]

            num_random_neg = 2 * self.num_hard_negative
            if num_random_neg < negative_loss.shape[0]:
                keep_index = random.sample(
                    range(int(negative_loss.shape[0])), num_random_neg
                )
                keep_index = torch.tensor(keep_index, device=ce_loss.device)
                negative_loss = negative_loss[keep_index]

        sorted_positive_loss, sorted_positive_index = torch.sort(
            positive_loss, descending=True
        )
        num_positive_keep = min(sorted_positive_loss.shape[0], self.num_hard_positive)
        if num_positive_keep <= 0:
            pass
        elif num_positive_keep < sorted_positive_loss.shape[0]:
            keep_pos_index = sorted_positive_index[:num_positive_keep]
            sorted_positive_loss = sorted_positive_loss[keep_pos_index]

        sorted_negative_loss, sorted_negative_index = torch.sort(
            negative_loss, descending=True
        )
        num_negative_keep = min(sorted_negative_loss.shape[0], self.num_hard_negative)
        if num_negative_keep <= 0:
            pass
        elif num_negative_keep < sorted_negative_loss.shape[0]:
            keep_neg_index = sorted_negative_index[:num_negative_keep]
            sorted_negative_loss = sorted_negative_loss[keep_neg_index]

        if self.reduction == "sum":
            keep_ce_loss = sorted_positive_loss.sum() + sorted_negative_loss.sum()
        elif self.reduction == "mean":
            keep_ce_loss = (sorted_positive_loss.sum() + sorted_negative_loss.sum()) / (
                num_positive_keep + num_negative_keep
            )
        elif self.reduction == "none":
            keep_ce_loss = torch.stack([sorted_positive_loss, sorted_negative_loss])
        else:
            raise ValueError(
                f"the given reduction value {self.reduction} is invalid, must be 'none', 'mean' or 'sum' "
            )

        return keep_ce_loss
