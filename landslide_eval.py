# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import torch
import fire
import time
import json

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from lavin.eval_model import ModelArgs, Transformer
from lavin.tokenizer import Tokenizer
from lavin.generator import LaVIN_Generator
from lavin.mm_adapter import set_MMAdapter, set_Clip_Adapter
from dataclasses import dataclass

from pathlib import Path
import fairscale.nn.model_parallel.initialize as fs_init
import torch.distributed as dist

from landslide_train import TrainArgs
from util.datasets import MSDataSet


@dataclass
class EvalArgs(TrainArgs):
    generation_temperature: float = 0.1
    top_p: float = 0.75


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def _load_and_redistribute_checkpoint(llama_model_path, model_name):

    with open(Path(llama_model_path) / model_name / 'params.json') as f:
        params = json.load(f)
    tokenizer = Tokenizer(model_path=str(Path(llama_model_path) / 'tokenizer.model'))
    print('Using model path: %s, model_name: %s' % (llama_model_path, model_name))

    checkpoints = (Path(llama_model_path) / model_name).glob('*.pth')
    checkpoints = sorted(checkpoints)

    if len(checkpoints) == 1:
        checkpoint = torch.load(llama_model_path + model_name + '/consolidated.00.pth', map_location="cpu")
        return checkpoint, tokenizer, params

    mp_world_size = fs_init.get_model_parallel_world_size()
    mp_rank = fs_init.get_model_parallel_rank()
    if mp_world_size == len(checkpoints):
        print('same number of shards of checkpoints and training, loading directly...')
        dist.barrier()
        print('[rank=%d, mp_rank=%d] loading from %s' % (dist.get_rank(), mp_rank, checkpoints[mp_rank]))
        checkpoint = torch.load(checkpoints[mp_rank], map_location='cpu')
    else:
        print('different number of shards of checkpoints and training, redistributing...')
        if dist.get_rank() == 0:
            loaded = []
            for x in checkpoints:
                print('loading from', x)
                loaded.append(torch.load(x, map_location='cpu'))

            full_state_dict = {}
            split_dims = {}

            def add_weight_with_split_dim(name, dim):
                if dim < 0:  # bcast without split
                    full_state_dict[name] = loaded[0][name].clone()
                else:
                    full_state_dict[name] = torch.cat([x[name] for x in loaded], dim=dim)
                for x in loaded:
                    del x[name]
                split_dims[name] = dim

            add_weight_with_split_dim('tok_embeddings.weight', 1)
            add_weight_with_split_dim('norm.weight', -1)
            add_weight_with_split_dim('output.weight', 0)
            for i in range(params['n_layers']):
                print('gathering layer %d of %d' % (i, params['n_layers']))
                layer_prefix = f'layers.{i}.'
                bcast_names = [
                    'attention_norm.weight',
                    'ffn_norm.weight',
                ]
                column_parallel_names = [
                    'attention.wq.weight',
                    'attention.wk.weight',
                    'attention.wv.weight',
                    'feed_forward.w1.weight',
                    'feed_forward.w3.weight',
                ]
                row_parallel_names = [
                    'attention.wo.weight',
                    'feed_forward.w2.weight',
                ]
                for key in bcast_names:
                    add_weight_with_split_dim(layer_prefix + key, -1)
                for key in column_parallel_names:
                    add_weight_with_split_dim(layer_prefix + key, 0)
                for key in row_parallel_names:
                    add_weight_with_split_dim(layer_prefix + key, 1)

            full_state_dict_meta = dict((k, v.shape) for k, v in full_state_dict.items())
            dist.broadcast_object_list([full_state_dict_meta, split_dims], src=0)

        else:  # dist.get_rank() != 0
            recv_objs = [None, None]
            dist.broadcast_object_list(recv_objs, src=0)
            full_state_dict_meta, split_dims = recv_objs

        local_state_dict = {}
        for k in sorted(full_state_dict_meta.keys()):
            print('redistributing weights: %s' % k)
            if dist.get_rank() == 0:
                value = full_state_dict[k].cuda().half()
                del full_state_dict[k]
            else:
                value = torch.empty(full_state_dict_meta[k], device='cuda', dtype=torch.half)
            dist.broadcast(value, src=0)
            value = value.cpu()
            if split_dims[k] < 0:
                local_state_dict[k] = value
            else:
                dim = split_dims[k]
                assert dim >= 0 and dim < value.ndim and value.size(dim) % mp_world_size == 0
                shard_size = value.size(dim) // mp_world_size
                shard_st, shard_ed = shard_size * mp_rank, shard_size * (mp_rank + 1)
                # TODO: make more general
                if dim == 0:
                    value = value[shard_st:shard_ed]
                elif dim == 1:
                    value = value[:, shard_st:shard_ed]
                else:
                    raise NotImplementedError()
                local_state_dict[k] = value.clone()

        checkpoint = local_state_dict

    return checkpoint, tokenizer, params


def load(args) -> LaVIN_Generator:
    start_time = time.time()
    checkpoint, tokenizer, params = _load_and_redistribute_checkpoint(args.llama_model_path, args.llm_model)

    if os.path.exists(args.adapter_path):
        print("Loading adapter checkpoint...")
        adapter_checkpoint = torch.load(args.adapter_path, map_location="cpu")
    else:
        print(f"Adapter checkpoint not found at {args.adapter_path}")

    model_args: ModelArgs = ModelArgs(max_seq_len=args.max_seq_len, hidden_proj=args.hidden_proj, **params)
    model_args.vocab_size = tokenizer.n_words

    if model_args.precision == 'bf16':
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    elif model_args.precision == 'fp16':
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    model = Transformer(model_args)

    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    t, s = args.temperature, args.adapter_scale
    set_MMAdapter(model, args.adapter_type, dim=args.adapter_dim, s=s, t=args.temperature, precision=model_args.precision)
    set_Clip_Adapter(model.backbone.visual, args.visual_adapter_type, dim=args.adapter_dim, s=s, t=t, precision='fp16')

    if os.path.exists(args.adapter_path):
        state_dict = {}
        for key in adapter_checkpoint['model']:
            state_dict[key.replace('module.', '')] = adapter_checkpoint['model'][key]
        model.load_state_dict(state_dict, strict=False)

    model = model.cuda()

    generator = LaVIN_Generator(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(**kwargs):
    args = EvalArgs(**kwargs)

    setup_model_parallel()
    generator = load(args)
    dataset = MSDataSet(args, 'all', args.llama_model_path, args.max_seq_len, test=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    for (images, indicators, prompt_questions, prompt_answers, idxs) in dataloader:
        _, predictions = generator.generate(
            prompt_questions,
            images=images,
            indicators=indicators,
            max_gen_len=args.max_seq_len,
            temperature=args.generation_temperature,
            n_feats=args.n_prompt,
            top_p=args.top_p,
        )

        print("Pred:", predictions)
        print("GT:", prompt_answers)


if __name__ == "__main__":
    fire.Fire(main)
