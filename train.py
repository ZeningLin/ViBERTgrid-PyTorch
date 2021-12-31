import os
import sys
import argparse
import yaml
import time

import torch
from transformers import BertTokenizer

from data.SROIE_dataset import load_train_dataset_multi_gpu
from model.ViBERTgrid_net import ViBERTgridNet
from pipeline.train_val_utils import (
    train_one_epoch,
    validate,
    cosine_scheduler,
    TensorboardLogger,
    TerminalLogger
)
from pipeline.distributed_utils import (
    init_distributed_mode,
    setup_seed,
    is_main_process,
    save_on_master,
)


def train(args):
    init_distributed_mode(args)
    setup_seed(42)

    with open(args.config_path, "r") as c:
        hyp = yaml.load(c, Loader=yaml.FullLoader)

    device = hyp["device"]
    start_epoch = hyp["start_epoch"]
    end_epoch = hyp["end_epoch"]
    batch_size = hyp["batch_size"]
    learning_rate = hyp["learning_rate"]
    min_learning_rate = hyp["min_learning_rate"]
    warm_up_epoches = hyp["warm_up_epoches"]
    warm_up_init_lr = hyp["warm_up_init_lr"]
    momentum = hyp["momentum"]
    weight_decay = hyp["weight_decay"]
    min_weight_decay = hyp["min_weight_decay"]

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
    roi_align_output_reshape = hyp["roi_align_output_reshape"]
    late_fusion_fuse_embedding_channel = hyp["late_fusion_fuse_embedding_channel"]
    loss_weights = hyp["loss_weights"]
    loss_control_lambda = hyp["loss_control_lambda"]

    device = torch.device(device)

    print(f"==> loading tokenizer {bert_version}")
    tokenizer = BertTokenizer.from_pretrained(bert_version)
    print(f"==> tokenizer {bert_version} loaded")

    print(f"==> loading datasets")
    train_loader, val_loader, train_sampler = load_train_dataset_multi_gpu(
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
        roi_align_output_reshape=roi_align_output_reshape,
        late_fusion_fuse_embedding_channel=late_fusion_fuse_embedding_channel,
        loss_weights=loss_weights,
        loss_control_lambda=loss_control_lambda,
    )
    model = model.to(device)
    model_wo_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_wo_ddp = model.module
    print(f"==> model created")

    num_training_steps_per_epoch = len(train_loader) // args.world_size
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params=params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
    )
    scaler = torch.cuda.amp.GradScaler() if amp else None

    if weights != "":
        print("==> loading pretrained")
        checkpoint = torch.load(weights, map_location="cpu")
        model_weights = {
            k.replace("module.", ""): v
            for k, v in checkpoint["model"].items()
        }
        model_wo_ddp.load_state_dict(model_weights)
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_schedule_values = (checkpoint["lr_scheduler"]).numpy()
        wd_schedule_values = (checkpoint["wd_scheduler"]).numpy()
        start_epoch = checkpoint["epoch"] + 1
        if amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print(f"==> pretrained {start_epoch} success")
    else:
        print("==> no pretrained")

    lr_schedule_values = cosine_scheduler(
        base_value=learning_rate,
        final_value=min_learning_rate,
        epoches=end_epoch,
        niter_per_ep=num_training_steps_per_epoch,
        warmup_epochs=warm_up_epoches,
        start_warmup_value=warm_up_init_lr,
        warmup_steps=-1,
    )
    wd_schedule_values = cosine_scheduler(
        base_value=weight_decay,
        final_value=min_weight_decay,
        epoches=end_epoch,
        niter_per_ep=num_training_steps_per_epoch,
    )

    logger = None
    if is_main_process():
        curr_time = time.localtime()
        curr_time_h = (
            f"{curr_time.tm_year:04d}-{curr_time.tm_mon:02d}-{curr_time.tm_mday:02d}"
        )
        curr_time_h += (
            f"_{curr_time.tm_hour:02d}:{curr_time.tm_min:02d}:{curr_time.tm_sec:02d}"
        )
        comment = f"{batch_size}_{learning_rate}_{curr_time_h}"
        logger = TensorboardLogger(comment=comment)
        if save_log != '':
            if not os.path.exists(save_log):
                os.mkdir(save_log)
            sys.stdout = TerminalLogger(os.path.join(save_log, comment + '.log'), sys.stdout)
            sys.stderr = TerminalLogger(os.path.join(save_log, comment + '.log'), sys.stdout)

    top_acc = 0.99
    top_F1 = 0.90
    print(f"==> start training")
    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if logger is not None:
            logger.set_step(epoch * num_training_steps_per_epoch)

        print(f"==> training epoch {epoch}/{end_epoch - start_epoch}")
        loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            epoch=epoch,
            start_step=epoch * num_training_steps_per_epoch,
            lr_scheduler=lr_schedule_values,
            weight_decay_scheduler=wd_schedule_values,
            logger=logger,
            scaler=scaler,
        )

        print(f"==> validating epoch {epoch}/{end_epoch - start_epoch}")
        classification_acc, F1 = validate(
            model=model,
            validate_loader=val_loader,
            device=device,
            epoch=epoch,
            logger=logger,
        )

        if F1 > top_F1 or (epoch % 400 == 0 and epoch != start_epoch):
            top_F1 = F1
            if save_top is not None:
                if not os.path.exists(save_top) and is_main_process():
                    os.mkdir(save_top)
                print(
                    f"==> top criteria found, saving model |epoch[{epoch}]|F1[{top_F1}]|acc[{classification_acc}]|"
                )
                save_files = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": torch.from_numpy(lr_schedule_values),
                    "weight_decay_scheduler": torch.from_numpy(wd_schedule_values),
                    "args": args,
                    "epoch": epoch,
                }
                if amp:
                    save_files["scaler"] = scaler.state_dict()

                curr_time = time.localtime()
                curr_time_h = (
                    f"{curr_time.tm_year:04d}-{curr_time.tm_mon:02d}-{curr_time.tm_mday:02d}"
                )
                curr_time_h += (
                    f"_{curr_time.tm_hour:02d}:{curr_time.tm_min:02d}:{curr_time.tm_sec:02d}"
                )

                save_on_master(
                    save_files,
                    os.path.join(
                        save_top,
                        f"{batch_size}_{learning_rate}_{backbone}_{bert_version}_{epoch}_{F1}_{curr_time_h}.pth",
                    ),
                )

        if is_main_process():
            if logger is not None:
                logger.flush()


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
