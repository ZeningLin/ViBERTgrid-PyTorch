import os
import sys
import argparse
import yaml
import time
import numpy as np

import torch
from transformers import BertTokenizer, RobertaTokenizer

from data.EPHOIE_dataset import load_train_dataset_multi_gpu as EPHOIE_load_train
from model.ViBERTgrid_net import ViBERTgridNet
from pipeline.train_val_utils import (
    train_one_epoch,
    validate,
    cosine_scheduler,
    TensorboardLogger,
    TerminalLogger,
)
from pipeline.distributed_utils import (
    init_distributed_mode,
    setup_seed,
    is_main_process,
    save_on_master,
)

EPHOIE_CLASS_LIST = [
    "其他",
    "年级",
    "科目",
    "学校",
    "考试时间",
    "班级",
    "姓名",
    "考号",
    "分数",
    "座号",
    "学号",
    "准考证号",
]

TAG_TO_IDX = {
    "O": 0,
    "B-grade": 1,
    "B-subject": 2,
    "B-school": 3,
    "B-testtime": 4,
    "B-class": 5,
    "B-name": 6,
    "B-testno": 7,
    "B-score": 8,
    "B-seatno": 9,
    "B-studentno": 10,
    "B-testadmissionno": 11,
}

TAG_TO_IDX_BIO = {
    "O": 0,
    "B-grade": 1,
    "I-grade": 2,
    "B-subject": 3,
    "I-subject": 4,
    "B-school": 5,
    "I-school": 6,
    "B-testtime": 7,
    "I-testtime": 8,
    "B-class": 9,
    "I-class": 10,
    "B-name": 11,
    "I-name": 12,
    "B-testno": 13,
    "I-testno": 14,
    "B-score": 15,
    "I-score": 16,
    "B-seatno": 17,
    "I-seatno": 18,
    "B-studentno": 19,
    "I-studentno": 20,
    "B-testadmissionno": 21,
    "I-testadmissionno": 22,
}


def train(args):
    init_distributed_mode(args)
    setup_seed(42)

    with open(args.config_path, "r") as c:
        hyp = yaml.load(c, Loader=yaml.FullLoader)

    comment_exp = hyp["comment"]

    device = hyp["device"]
    sync_bn = hyp["syncBN"]

    start_epoch = hyp["start_epoch"]
    end_epoch = hyp["end_epoch"]
    batch_size = hyp["batch_size"]

    learning_rate_cnn = hyp["optimizer_cnn_hyp"]["learning_rate"]
    min_learning_rate_cnn = hyp["optimizer_cnn_hyp"]["min_learning_rate"]
    warm_up_epoches_cnn = hyp["optimizer_cnn_hyp"]["warm_up_epoches"]
    warm_up_init_lr_cnn = hyp["optimizer_cnn_hyp"]["warm_up_init_lr"]
    momentum_cnn = hyp["optimizer_cnn_hyp"]["momentum"]
    weight_decay_cnn = hyp["optimizer_cnn_hyp"]["weight_decay"]
    min_weight_decay_cnn = hyp["optimizer_cnn_hyp"]["min_weight_decay"]

    learning_rate_bert = hyp["optimizer_bert_hyp"]["learning_rate"]
    min_learning_rate_bert = hyp["optimizer_bert_hyp"]["min_learning_rate"]
    warm_up_epoches_bert = hyp["optimizer_bert_hyp"]["warm_up_epoches"]
    warm_up_init_lr_bert = hyp["optimizer_bert_hyp"]["warm_up_init_lr"]
    beta_1_bert = hyp["optimizer_bert_hyp"]["beta1"]
    beta_2_bert = hyp["optimizer_bert_hyp"]["beta2"]
    epsilon_bert = hyp["optimizer_bert_hyp"]["epsilon"]
    weight_decay_bert = hyp["optimizer_bert_hyp"]["weight_decay"]
    min_weight_decay_bert = hyp["optimizer_bert_hyp"]["min_weight_decay"]

    num_hard_positive_main_1 = hyp["num_hard_positive_main_1"]
    num_hard_negative_main_1 = hyp["num_hard_negative_main_1"]
    num_hard_positive_main_2 = hyp["num_hard_positive_main_2"]
    num_hard_negative_main_2 = hyp["num_hard_negative_main_2"]
    loss_aux_sample_list = hyp["loss_aux_sample_list"]
    num_hard_positive_aux = hyp["num_hard_positive_aux"]
    num_hard_negative_aux = hyp["num_hard_negative_aux"]
    ohem_random = hyp["ohem_random"]

    save_top = hyp["save_top"]
    save_log = hyp["save_log"]

    amp = hyp["amp"]
    weights = hyp["weights"]

    num_workers = hyp["num_workers"]
    data_root = hyp["data_root"]
    num_classes = hyp["num_classes"]
    image_mean = hyp["image_mean"]
    image_std = hyp["image_std"]
    image_min_size = hyp["image_min_size"]
    image_max_size = hyp["image_max_size"]
    test_image_min_size = hyp["test_image_min_size"]

    bert_version = hyp["bert_version"]
    backbone = hyp["backbone"]
    grid_mode = hyp["grid_mode"]
    early_fusion_downsampling_ratio = hyp["early_fusion_downsampling_ratio"]
    roi_shape = hyp["roi_shape"]
    p_fuse_downsampling_ratio = hyp["p_fuse_downsampling_ratio"]
    late_fusion_fuse_embedding_channel = hyp["late_fusion_fuse_embedding_channel"]
    loss_weights = hyp["loss_weights"]
    loss_control_lambda = hyp["loss_control_lambda"]
    layer_mode = hyp["layer_mode"]

    classifier_mode = hyp["classifier_mode"]
    eval_mode = hyp["eval_mode"]
    tag_mode = hyp["tag_mode"]

    if tag_mode == "BIO":
        map_dict = TAG_TO_IDX_BIO
    else:
        map_dict = TAG_TO_IDX

    device = torch.device(device)

    print(f"==> loading tokenizer {bert_version}")
    if "bert-" in bert_version:
        tokenizer = BertTokenizer.from_pretrained(bert_version)
    elif "roberta-" in bert_version:
        tokenizer = RobertaTokenizer.from_pretrained(bert_version)
    print(f"==> tokenizer {bert_version} loaded")

    print(f"==> loading datasets")
    train_loader, val_loader, train_sampler = EPHOIE_load_train(
        root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        tokenizer=tokenizer,
    )
    print(f"==> dataset loaded")

    print(f"==> creating model {backbone} | {bert_version}")
    model = ViBERTgridNet(
        num_classes=num_classes,
        image_mean=image_mean,
        image_std=image_std,
        image_min_size=image_min_size,
        image_max_size=image_max_size,
        test_image_min_size=test_image_min_size,
        bert_model=bert_version,
        tokenizer=tokenizer,
        backbone=backbone,
        grid_mode=grid_mode,
        early_fusion_downsampling_ratio=early_fusion_downsampling_ratio,
        roi_shape=roi_shape,
        p_fuse_downsampling_ratio=p_fuse_downsampling_ratio,
        late_fusion_fuse_embedding_channel=late_fusion_fuse_embedding_channel,
        loss_weights=loss_weights,
        loss_control_lambda=loss_control_lambda,
        num_hard_positive_main_1=num_hard_positive_main_1,
        num_hard_negative_main_1=num_hard_negative_main_1,
        num_hard_positive_main_2=num_hard_positive_main_2,
        num_hard_negative_main_2=num_hard_negative_main_2,
        num_hard_positive_aux=num_hard_positive_aux,
        num_hard_negative_aux=num_hard_negative_aux,
        loss_aux_sample_list=loss_aux_sample_list,
        classifier_mode=classifier_mode,
        tag_to_idx=map_dict,
        ohem_random=ohem_random,
        layer_mode=layer_mode,
    )
    if sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model_wo_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_wo_ddp = model.module
    print(f"==> model created")

    num_training_steps_per_epoch = len(train_loader) // args.world_size

    params_cnn = []
    params_bert = []
    for name, parameters in model.named_parameters():
        if "bert_model" in name and parameters.requires_grad:
            params_bert.append(parameters)
        elif parameters.requires_grad:
            params_cnn.append(parameters)

    optimizer_cnn = torch.optim.SGD(
        params=params_cnn,
        lr=learning_rate_cnn,
        momentum=momentum_cnn,
        weight_decay=weight_decay_cnn,
    )
    optimizer_bert = torch.optim.AdamW(
        params=params_bert,
        lr=learning_rate_bert,
        betas=(beta_1_bert, beta_2_bert),
        eps=epsilon_bert,
        weight_decay=weight_decay_bert,
    )

    scaler = torch.cuda.amp.GradScaler() if amp else None

    lr_schedule_values_cnn = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer_cnn, step_size=15, gamma=0.1
    )
    wd_schedule_values_cnn = cosine_scheduler(
        base_value=weight_decay_cnn,
        final_value=min_weight_decay_cnn,
        epoches=end_epoch,
        niter_per_ep=num_training_steps_per_epoch,
    )

    lr_schedule_values_bert = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer_bert, step_size=15, gamma=0.1
    )
    wd_schedule_values_bert = cosine_scheduler(
        base_value=weight_decay_bert,
        final_value=min_weight_decay_bert,
        epoches=end_epoch,
        niter_per_ep=num_training_steps_per_epoch,
    )

    if weights != "":
        print("==> loading pretrained")
        checkpoint = torch.load(weights, map_location="cpu")
        model_wo_ddp.load_state_dict(checkpoint["model"])
        optimizer_cnn.load_state_dict(checkpoint["optimizer_cnn"])
        optimizer_bert.load_state_dict(checkpoint["optimizer_bert"])
        if isinstance(lr_schedule_values_cnn, np.ndarray):
            lr_schedule_values_cnn = (checkpoint["lr_scheduler_cnn"]).numpy()
        else:
            lr_schedule_values_cnn.load_state_dict(checkpoint["lr_scheduler_cnn"])
        wd_schedule_values_cnn = (checkpoint["wd_scheduler_cnn"]).numpy()
        if isinstance(lr_schedule_values_bert, np.ndarray):
            lr_schedule_values_bert = (checkpoint["lr_scheduler_bert"]).numpy()
        else:
            lr_schedule_values_bert.load_state_dict(checkpoint["lr_scheduler_bert"])
        wd_schedule_values_bert = (checkpoint["wd_scheduler_bert"]).numpy()
        start_epoch = checkpoint["epoch"] + 1
        if amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print(f"==> pretrained {start_epoch} success")
    else:
        print("==> no pretrained")

    logger = None
    if is_main_process():
        curr_time = time.localtime()
        curr_time_h = (
            f"{curr_time.tm_year:04d}-{curr_time.tm_mon:02d}-{curr_time.tm_mday:02d}"
        )
        curr_time_h += (
            f"_{curr_time.tm_hour:02d}:{curr_time.tm_min:02d}:{curr_time.tm_sec:02d}"
        )
        comment = (
            comment_exp
            + f"bb-{backbone}_bertv-{bert_version.replace('/', '_')}_bs-{batch_size}"
            f"_lr1-{learning_rate_cnn}_lr2-{learning_rate_bert}_time-{curr_time_h}"
        )
        logger = TensorboardLogger(comment=comment)
        if save_log != "":
            if not os.path.exists(save_log):
                os.mkdir(save_log)
            sys.stdout = TerminalLogger(
                os.path.join(save_log, comment + ".log"), sys.stdout
            )
            sys.stderr = TerminalLogger(
                os.path.join(save_log, comment + ".log"), sys.stdout
            )

    print(f"==> Initial validation")
    F1 = validate(
        model=model,
        validate_loader=val_loader,
        device=device,
        epoch=0,
        logger=logger,
        eval_mode=eval_mode,
        tag_to_idx=map_dict,
        category_list=EPHOIE_CLASS_LIST,
        language="chn"
    )

    top_F1_tresh = 0.97
    top_F1 = 0
    print(f"==> start training")
    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if logger is not None:
            logger.set_step(epoch * num_training_steps_per_epoch)

        print(f"==> training epoch {epoch + 1}/{end_epoch - start_epoch}")
        loss = train_one_epoch(
            model=model,
            optimizer_cnn=optimizer_cnn,
            optimizer_bert=optimizer_bert,
            train_loader=train_loader,
            device=device,
            epoch=epoch,
            start_step=epoch * num_training_steps_per_epoch,
            lr_scheduler_cnn=lr_schedule_values_cnn,
            weight_decay_scheduler_cnn=wd_schedule_values_cnn,
            lr_scheduler_bert=lr_schedule_values_bert,
            weight_decay_scheduler_bert=wd_schedule_values_bert,
            logger=logger,
            scaler=scaler,
        )

        print(f"==> validating epoch {epoch + 1}/{end_epoch - start_epoch}")
        F1 = validate(
            model=model,
            validate_loader=val_loader,
            device=device,
            epoch=epoch,
            logger=logger,
            eval_mode=eval_mode,
            tag_to_idx=map_dict,
            category_list=EPHOIE_CLASS_LIST,
            language="chn"
        )

        if F1 > top_F1:
            top_F1 = F1

        if F1 > top_F1_tresh or (epoch % 400 == 0 and epoch != start_epoch):
            top_F1_tresh = F1
            if save_top is not None:
                if not os.path.exists(save_top) and is_main_process():
                    os.mkdir(save_top)
                print(f"==> top criteria found, saving model |epoch[{epoch}]|F1[{F1}]|")
                save_files = {
                    "model": model.state_dict(),
                    "optimizer_cnn": optimizer_cnn.state_dict(),
                    "optimizer_bert": optimizer_bert.state_dict(),
                    "lr_scheduler_cnn": torch.from_numpy(lr_schedule_values_cnn)
                    if isinstance(lr_schedule_values_cnn, np.ndarray)
                    else lr_schedule_values_cnn.state_dict(),
                    "weight_decay_scheduler_cnn": torch.from_numpy(
                        wd_schedule_values_cnn
                    ),
                    "lr_scheduler_bert": torch.from_numpy(lr_schedule_values_bert)
                    if isinstance(lr_schedule_values_bert, np.ndarray)
                    else lr_schedule_values_bert.state_dict(),
                    "weight_decay_scheduler_bert": torch.from_numpy(
                        wd_schedule_values_bert
                    ),
                    "args": args,
                    "epoch": epoch,
                }
                if amp:
                    save_files["scaler"] = scaler.state_dict()

                curr_time = time.localtime()
                curr_time_h = f"{curr_time.tm_year:04d}-{curr_time.tm_mon:02d}-{curr_time.tm_mday:02d}"
                curr_time_h += f"_{curr_time.tm_hour:02d}:{curr_time.tm_min:02d}:{curr_time.tm_sec:02d}"

                save_on_master(
                    save_files,
                    os.path.join(
                        save_top,
                        f"bs-{batch_size}_lr1-{learning_rate_cnn}_lr2-{learning_rate_bert}_"
                        f"bb-{backbone}_bertv-{bert_version.replace('/', '_')}_epoch-{epoch}_F1-{F1}_time-{curr_time_h}.pth",
                    ),
                )

        if is_main_process():
            if logger is not None:
                logger.flush()

    print(f"top_F1: {top_F1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        required=True,
        help="directory to the configuration yaml file",
    )
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )
    args = parser.parse_args()

    train(args)
