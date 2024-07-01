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

from pathlib import Path
import fairscale.nn.model_parallel.initialize as fs_init
import torch.distributed as dist

from landslide_train import TrainArgs
from util.datasets import MSDataSet
from openai import OpenAI
import wandb


def evaluate_similarity(prediction, answer):

    prompt = f"""
[Question] Describe the image in the aspects of disaster type, cause, detailed observations, and future risk

[Human]
{answer}
[End of Human]

[AI Assistant]
{prediction}
[End of AI Assistant]

We would like to request your feedback on the performance of AI Assistant if it can generate responses having the same meaning with the responses from Human. Please rate the similarity level of details of the responses from AI Assistant and Human. Rate the responses separately in four topics including "Disaster Type", "Cause", "Observation", and "Future risk". AI Assistant receives a score for each topic on a scale of 1 to 10, where a higher score indicates better overall performance. Please first output a single line containing only four values indicating the scores for AI Assistant in the topics of "Disaster Type", "Cause", "Observation", and "Future risk", respectively. The four scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your evaluation.
"""
    return prompt

    client = OpenAI(api_key="OPENAI_API_KEY")
    response = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": prompt,
        }],
        model="gpt-4",
        max_tokens=512,
        temperature=0,
    )
    return response.choices[0].text.strip()


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


def main(
    llama_model_path="./data/weights/",
    llm_model="7B",
    adapter_path='7B-checkpoint-99.pth',
    visual_adapter_type='normal',
    adapter_type='attn',
    temperature=0.1,
    top_p=0.75,
    sampling_seed=23,
    gpt_score=False,
):

    args = TrainArgs(
        llama_model_path=llama_model_path,
        llm_model=llm_model,
        visual_adapter_type=visual_adapter_type,
        adapter_type=adapter_type,
        adapter_path=adapter_path,
    )
    setup_model_parallel()
    generator = load(args)
    dataset = MSDataSet(args, 'all', args.llama_model_path, args.max_seq_len, test=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    adapter_name = adapter_path.split('/')[-1].split('.')[0]  # checkpoint-99
    dir_path = f"./models/quang/results/{llm_model}-ft_{adapter_name}"
    exp_name = f"temp{temperature}-top_p{top_p}_seed{sampling_seed}"
    log_path = f"{dir_path}/log_{exp_name}.txt"
    gpt_path = f"{dir_path}/gpt_score.txt"

    wandb.init(
        project='landslide',
        entity="landslide_tohoku",
        name=f"{llm_model}-ft_{adapter_name}-{exp_name}",
    )

    os.makedirs(dir_path, exist_ok=True)

    if gpt_score:
        with open(gpt_path, "a") as f:
            f.write(f"\n\nexp_name: {exp_name}\n")

    avg_scores = []
    for (images, _, _, prompt_answers, image_paths) in dataloader:
        _, predictions = generator.generate(
            prompts=[
                """Instruction:  Describe the image in the aspects of disaster type, cause, detailed observations, and future risk using the following template. Template: { Type : [TXT], Cause : [TXT], Observation : [[TXT], [TXT], [TXT], ...], Future risk : [TXT] }.
Responese:"""
            ],
            images=images,
            indicators=[1],
            max_gen_len=512,
            n_feats=6,
            temperature=temperature,
            top_p=top_p,
            sampling_seed=sampling_seed,
        )
        prediction = predictions[0]
        answer = prompt_answers[0]

        with open(log_path, "a") as f:
            f.write(f"Image path: {image_paths[0]}\n")
            f.write(f"Pred: {prediction}\n")
            f.write(f"GT: {answer}\n")
            f.write("-----------------\n")

            # print to console
            # print("image path:", image_paths[0])
            # print("Pred:", prediction, "\n")
            # print("GT:", answer)
            # print("-----------------\n")

        gpt_feedback = evaluate_similarity(prediction, answer)
        print(gpt_feedback)
        print("\n\n")
        print("========================================\n\n")

        with open(log_path, "a") as f:
            f.write(gpt_feedback)
            f.write("\n")

        if False:
            gpt_feedback = evaluate_similarity(prediction, answer)
            with open(log_path, "a") as f:
                f.write(f"GPT-4: \n{gpt_feedback}\n")
                f.write("========================================\n\n")

                print("GPT-4: \n", gpt_feedback, "\n")
                print("========================================\n\n")

            # Extract the first line containing the scores
            first_line = gpt_feedback.split("\n")[0]

            # Split the scores and convert them to float
            scores = list(map(float, first_line.split()))  # [1.0, 2.0, 3.0, 4.0]
            avg_score = sum(scores) / len(scores)
            avg_scores.append(avg_score)

            with open(gpt_path, "a") as f:
                f.write(f"Image path: {image_paths[0]}, scores: {scores}, avg_score: {avg_score}\n")


if __name__ == "__main__":
    fire.Fire(main)
