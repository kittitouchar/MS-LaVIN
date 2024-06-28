from math import dist
import os
import argparse
import datetime
import json
from random import seed
import time
import fire

from click import prompt
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine import train_one_epoch

from util.datasets import ScienceQADataSet, InstrcutDataSet, MSDataSet, SumDataSet, ParaphraseDataSet
from lavin.mm_adaptation import LaVIN
import wandb
import time

@dataclass
class TrainArgs:
    batch_size: int = 2  # per gpu, effective batch size is batch_size * accum_iter * # gpus
    accum_iter: int = 2
    epochs: int = 100
    wandb_enable: bool = False

    # Model parameters
    llama_model_path: str = './data/weights_2/'
    llm_model: str = 'llama-2-13b-chat'  # 7B
    use_vicuna: bool = False

    visual_adapter_type: str = 'normal'  # router, router_block,
    adapter_type: str = 'attn'  # normal, attn, adapter_box
    adapter_dim: int = 8
    hidden_proj: int = 128  # the visual adapter dim
    temperature: float = 10.
    n_prompt: int = 6  # length of visual features
    adapter_scale: float = 1.  # the scales of adapter layer
    drop_path: float = 0.
    max_seq_len: int = 512

    # Optimizer parameters
    weight_decay: float = 0.05
    lr: float = None
    clip_grad: float = None
    blr: float = 1e-3
    min_lr: float = 0.
    gradient_checkpointing: bool = False
    warmup_epochs: float = 2

    start_epoch: int = 0
    num_workers: int = 2
    pin_mem: bool = True

    # Distributed training parameters
    device: str = 'cuda'
    dist_on_itp: bool = False
    distributed: bool = False
    dist_url: str = "env://"
    world_size: int = 1
    local_rank: int = 0
    rank: int = 0

    seed: int = 0
    resume: str = ''
    adapter_path: str = './checkpoint-99.pth'  # path to adapter checkpoint
    wandb_enable: bool = False

    # datasets
    data_path: str = '/instruction_dataset/'
    output_dir: str = './output_dir/test'  # path where to save, empty for no saving
    log_dir: str = './output_dir/test'  # path where to tensorboard log
    prompt_format: str = 'CQM-A'
    options: list = field(default_factory=lambda: ["A", "B", "C", "D", "E"])
    caption_file: str = './data/captions.json'
    data_root: str = '../data'
    use_caption: bool = False  # use image captions or not
    do_finetune: bool = False  # pre-train on large scale vl instruction
    do_pretrain: bool = False  # pre-train on large scale vl instruction
    do_sum: bool = False  # finetune on summarization task
    do_paraphrase: bool = False  # finetune on paraphrase task


def main(**kwargs):
    args = TrainArgs(**kwargs)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), mode="w", encoding="utf-8") as f:
        f.write(json.dumps(asdict(args), indent=4, sort_keys=True))

    misc.init_distributed_mode(args)

    if misc.is_main_process() and args.wandb_enable:
        wandb.init(project='landslide', 
        entity="landslide_tohoku",
        name=args.output_dir.split('/')[-1],
        dir=args.output_dir,
        config=vars(args),
        )


    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    ## choose dataset by do_x
    if args.do_pretrain:
        dataset_train = InstrcutDataSet(args, 'all', args.llama_model_path, args.max_seq_len)

    elif args.do_finetune:
        dataset_train = MSDataSet(args, 'all', args.llama_model_path, args.max_seq_len)

    elif args.do_sum:
        dataset_train = SumDataSet(args, 'all', args.llama_model_path, args.max_seq_len)

    elif args.do_paraphrase:
        dataset_train = ParaphraseDataSet(args, 'all', args.llama_model_path, args.max_seq_len)

    else:
        dataset_train = ScienceQADataSet(args, 'train', args.llama_model_path, args.max_seq_len)

    print(dataset_train)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)

    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define the model
    model = LaVIN(args)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    print(args.batch_size, args.accum_iter, misc.get_world_size())
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        print(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu]), find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, log_writer=log_writer, args=args)

        if args.output_dir:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {
            **{
                f'train_{k}': v for k, v in train_stats.items()
            },
            'epoch': epoch,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    fire.Fire(main)
