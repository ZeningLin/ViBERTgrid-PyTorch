import sys
import time
import math

import numpy as np

from typing import Iterable, Any, List

import torch
import torch.distributed
import torch.backends.cudnn

from pipeline.distributed_utils import reduce_loss, get_world_size
from pipeline.criteria import (
    SROIE_label_classification_criteria,
    SROIE_label_F1_criteria,
)
from utils.ViBERTgrid_visualize import inference_visualize


class TerminalLogger(object):
    def __init__(self, filename, stream=sys.stdout):
        super().__init__()
        self.terminal = stream
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class TensorboardLogger(object):
    def __init__(self, comment=None) -> None:
        super().__init__()
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(comment=comment)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head="scalar", step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(
                head + "/" + k, v, self.step if step is None else step
            )

    def flush(self):
        self.writer.flush()


def cosine_scheduler(
    base_value,
    final_value,
    epoches,
    niter_per_ep,
    warmup_epoches=0,
    start_warmup_value=0,
    warmup_steps=-1,
) -> np.ndarray:
    warmup_schedule = np.array([])
    warmup_iters = warmup_epoches * (niter_per_ep + 1)
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epoches > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epoches * (niter_per_ep + 1) - warmup_iters)
    schedule = np.array(
        [
            final_value
            + 0.5
            * (base_value - final_value)
            * (1 + math.cos(math.pi * i / (len(iters))))
            for i in iters
        ]
    )

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epoches * (niter_per_ep + 1)
    return schedule


def step_scheduler(
    base_value: float,
    steps: List,
    gamma: float,
    num_epoches: int,
    niter_per_ep: int,
    warmup_epoches: int = 0,
    start_warmup_value=0,
    warmup_steps=-1,
) -> np.ndarray:
    warmup_schedule = np.array([])
    warmup_iters = warmup_epoches * (niter_per_ep + 1)
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epoches > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    change_steps = [step * niter_per_ep for step in steps]
    change_steps.append(num_epoches * (niter_per_ep + 1))
    schedule = [warmup_schedule]
    curr_value = base_value
    start_step = warmup_iters
    for change_step in change_steps:
        end_step = change_step
        curr_schedule = curr_value * np.ones((end_step - start_step))
        schedule.append(curr_schedule)
        curr_value *= gamma
        start_step = end_step
    schedule = np.concatenate(schedule)
    assert len(schedule) == num_epoches * (niter_per_ep + 1)
    
    return schedule


def train_one_epoch(
    model: torch.nn.Module,
    optimizer_cnn: torch.optim.Optimizer,
    optimizer_bert: torch.optim.Optimizer,
    train_loader: Iterable,
    device: torch.device,
    epoch: int,
    start_step: int,
    lr_scheduler_cnn: Any,
    weight_decay_scheduler_cnn: Any,
    lr_scheduler_bert: Any,
    weight_decay_scheduler_bert: Any,
    logger: TensorboardLogger = None,
    scaler: torch.cuda.amp.GradScaler = None,
):
    assert isinstance(
        lr_scheduler_cnn,
        (
            np.ndarray,
            torch.optim.lr_scheduler.StepLR,
            torch.optim.lr_scheduler.MultiStepLR,
        ),
    ), f"invalid lr_scheduler_cnn, must be numpy.ndarray or torch.optim.lr_scheduler, {type(lr_scheduler_cnn)} given"
    assert isinstance(
        lr_scheduler_bert,
        (
            np.ndarray,
            torch.optim.lr_scheduler.StepLR,
            torch.optim.lr_scheduler.MultiStepLR,
        ),
    ), f"invalid lr_scheduler_cnn, must be numpy.ndarray or torch.optim.lr_scheduler, {type(lr_scheduler_bert)} given"

    start_time = time.time()

    MB = 1024.0 * 1024.0
    total_iter = str(len(train_loader))
    if torch.cuda.is_available():
        log_message = "  ".join(
            [
                "\t",
                "epoch[{epoch}]",
                "iter[{iter}]/[" + total_iter + "]",
                "train_loss: {train_loss:.4f}",
                "time used: {iter_time:.0f}s",
                "max mem: {memory:.0f}",
            ]
        )
    else:
        log_message = "  ".join(
            [
                "\t",
                "epoch[{epoch}]",
                "iter[{iter}]/[" + total_iter + "]",
                "train_loss: {train_loss:.4f}",
                "time used: {iter_time:.0f}s",
            ]
        )

    model.train()

    mean_train_loss = torch.zeros(1).to(device)
    for step, train_batch in enumerate(train_loader):
        iter_ = start_step + step
        if isinstance(lr_scheduler_cnn, np.ndarray):
            for param_group in optimizer_cnn.param_groups:
                if lr_scheduler_cnn is not None:
                    param_group["lr"] = lr_scheduler_cnn[iter_]
                if (
                    weight_decay_scheduler_cnn is not None
                    and param_group["weight_decay"] > 0
                ):
                    param_group["weight_decay"] = weight_decay_scheduler_cnn[iter_]
        if isinstance(lr_scheduler_bert, np.ndarray):
            for param_group in optimizer_cnn.param_groups:
                if lr_scheduler_bert is not None:
                    param_group["lr"] = lr_scheduler_bert[iter_]
                if (
                    weight_decay_scheduler_bert is not None
                    and param_group["weight_decay"] > 0
                ):
                    param_group["weight_decay"] = weight_decay_scheduler_bert[iter_]

        (
            image_list,
            class_labels,
            pos_neg_labels,
            ocr_coors,
            ocr_corpus,
            mask,
        ) = train_batch

        image_list = tuple(image.to(device) for image in image_list)
        class_labels = tuple(class_label.to(device) for class_label in class_labels)
        pos_neg_labels = tuple(
            pos_neg_label.to(device) for pos_neg_label in pos_neg_labels
        )
        ocr_coors = ocr_coors.to(device)
        ocr_corpus = ocr_corpus.to(device)
        mask = mask.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            train_loss = model(
                image_list, class_labels, pos_neg_labels, ocr_coors, ocr_corpus, mask
            )

        # torch.distributed.barrier()
        # train_loss = reduce_loss(train_loss)
        train_loss_value = train_loss.item()
        mean_train_loss = (mean_train_loss * step + train_loss_value) / (step + 1)

        if not math.isfinite(train_loss_value):
            print(f"loss is {train_loss_value}, training will stop")
            sys.exit(1)

        optimizer_cnn.zero_grad()
        optimizer_bert.zero_grad()
        if scaler is not None:
            scaler.scale(train_loss).backward()
            scaler.step(optimizer_cnn)
            scaler.step(optimizer_bert)
            scaler.update()
        else:
            train_loss.backward()
            optimizer_cnn.step()
            optimizer_bert.step()

        end_time = time.time()
        time_iter = end_time - start_time
        start_time = end_time

        if torch.cuda.is_available():
            print(
                log_message.format(
                    epoch=epoch,
                    iter=step + 1,
                    train_loss=train_loss_value,
                    iter_time=time_iter,
                    memory=torch.cuda.max_memory_allocated() / MB,
                )
            )
        else:
            print(
                log_message.format(
                    epoch=epoch,
                    iter=step + 1,
                    train_loss=train_loss_value,
                    iter_time=time_iter,
                )
            )

        if logger is not None:
            logger.update(head="loss", train_loss=train_loss_value)
            logger.update(
                head="opt", weight_decay_cnn=weight_decay_scheduler_cnn[iter_]
            )
            logger.update(
                head="opt", weight_decay_bert=weight_decay_scheduler_bert[iter_]
            )
            if isinstance(lr_scheduler_cnn, np.ndarray):
                logger.update(head="opt", lr_cnn=lr_scheduler_cnn[iter_])
            else:
                logger.update(head="opt", lr_cnn=lr_scheduler_cnn.get_last_lr()[0])
            if isinstance(lr_scheduler_bert, np.ndarray):
                logger.update(head="opt", lr_bert=lr_scheduler_bert[iter_])
            else:
                logger.update(head="opt", lr_bert=lr_scheduler_bert.get_last_lr()[0])

            logger.set_step()

    if not isinstance(lr_scheduler_cnn, np.ndarray):
        lr_scheduler_cnn.step()
    if not isinstance(lr_scheduler_bert, np.ndarray):
        lr_scheduler_bert.step()

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return train_loss_value


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    validate_loader: Iterable,
    device: torch.device,
    epoch: int,
    logger: TensorboardLogger,
    distributed: bool = True,
    iter_msg: bool = True,
):
    num_iter = len(validate_loader)
    start_time = time.time()
    iter_message = " ".join(
        [
            "\t",
            "epoch[{epoch}]",
            "iter[{iter}]/[{num_iter}]",
        ]
    )
    log_message = " ".join(
        [
            "\t",
            "epoch[{epoch}]",
            "validate_loss: {val_loss}",
            "classification_acc: {acc:.4f}%",
            "precision: {precision:.4f}",
            "recall: {recall:.4f}",
            "F1: {F1:.4f}",
            "time used: {time_used:.0f}s",
            "\n",
        ]
    )

    model.eval()
    total_num_correct, total_num_entities = 0.0, 0.0
    num_precision, num_recall, num_pred_key, num_gt_key = 0.0, 0.0, 0.0, 0.0
    mean_validate_loss = torch.zeros(1).to(device)
    for step, validate_batch in enumerate(validate_loader):
        (
            image_list,
            class_labels,
            pos_neg_labels,
            ocr_coors,
            ocr_corpus,
            mask,
        ) = validate_batch

        image_list = tuple(image.to(device) for image in image_list)
        class_labels = tuple(class_label.to(device) for class_label in class_labels)
        pos_neg_labels = tuple(
            pos_neg_label.to(device) for pos_neg_label in pos_neg_labels
        )
        ocr_coors = ocr_coors.to(device)
        ocr_corpus = ocr_corpus.to(device)
        mask = mask.to(device)

        validate_loss, pred_mask, pred_ss, gt_label, pred_label = model(
            image_list, class_labels, pos_neg_labels, ocr_coors, ocr_corpus, mask
        )

        validate_loss = reduce_loss(validate_loss)
        validate_loss_value = validate_loss.item()
        mean_validate_loss = (mean_validate_loss * step + validate_loss_value) / (
            step + 1
        )

        if distributed:
            torch.distributed.barrier()

        num_correct, num_entities = SROIE_label_classification_criteria(
            gt_label=gt_label, pred_label=pred_label
        )

        (
            num_precision_,
            num_recall_,
            num_pred_key_,
            num_gt_key_,
        ) = SROIE_label_F1_criteria(gt_label=gt_label, pred_label=pred_label)

        num_precision += num_precision_
        num_recall += num_recall_
        num_pred_key += num_pred_key_
        num_gt_key += num_gt_key_

        total_num_correct += num_correct
        total_num_entities += num_entities

        if iter_msg:
            print(
                iter_message.format(
                    epoch=epoch,
                    iter=step + 1,
                    num_iter=num_iter,
                )
            )

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    acc = total_num_correct / (total_num_entities + 1e-8)
    precision = float(0) if (num_precision == 0) else (num_precision / num_pred_key)
    recall = float(1) if (num_recall == 0) else (num_recall / num_gt_key)
    F1 = (
        float(0)
        if (precision + recall == 0)
        else (2 * precision * recall) / (precision + recall)
    )

    time_used = time.time() - start_time
    print(
        log_message.format(
            epoch=epoch,
            val_loss=validate_loss_value,
            acc=acc,
            precision=precision,
            recall=recall,
            F1=F1,
            time_used=time_used,
        )
    )

    if logger is not None:
        logger.update(head="loss", validate_loss=validate_loss_value, step=epoch)
        logger.update(head="criteria", label_classification_acc=acc, step=epoch)
        logger.update(head="criteria", label_precision=precision, step=epoch)
        logger.update(head="criteria", label_recall=recall, step=epoch)
        logger.update(head="criteria", label_F1=F1, step=epoch)

    return acc, F1


@torch.no_grad()
def inference_once(
    model: torch.nn.Module,
    batch: tuple,
    device: torch.device,
):
    model.eval()

    (
        image_list,
        class_labels,
        pos_neg_labels,
        ocr_coors,
        ocr_corpus,
        mask,
    ) = batch

    image_list = tuple(image.to(device) for image in image_list)
    class_labels = tuple(class_label.to(device) for class_label in class_labels)
    pos_neg_labels = tuple(pos_neg_label.to(device) for pos_neg_label in pos_neg_labels)
    ocr_coors = ocr_coors.to(device)
    ocr_corpus = ocr_corpus.to(device)
    mask = mask.to(device)

    start_time = time.time()
    _, pred_mask, pred_ss, gt_label, pred_label = model(
        image_list, class_labels, pos_neg_labels, ocr_coors, ocr_corpus, mask
    )
    time_used = time.time() - start_time
    print(f"inference speed: {time_used * 1000}ms")

    inference_visualize(
        image=image_list[0],
        class_label=class_labels[0],
        pred_ss=pred_ss,
        pred_mask=pred_mask,
    )
