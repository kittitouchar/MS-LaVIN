import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
from PIL import Image
# from minigpt4.common.config import Config
from util.misc import get_rank
# from minigpt4.common.registry import registry
from conversation.conversation import Chat, CONV_VISION
from torchvision.transforms import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from eval import load
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from typing import Tuple
from landslide_eval import load
from landslide_train import TrainArgs


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--llama_model_path", type=str, default="./data/weights/", help="dir of pre-trained weights.")
    parser.add_argument("--llm_model", type=str, default="7B", help="the type of llm.")
    parser.add_argument('--adapter_path', type=str, default='7B-checkpoint-99.pth')
    args = parser.parse_args()

    eval_args = TrainArgs(llama_model_path=args.llama_model_path, adapter_path=args.adapter_path, llm_model=args.llm_model)
    return eval_args


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
eval_args = parse_args()

local_rank, world_size = setup_model_parallel()
lavin = load(eval_args)
vis_processor = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])
chat = Chat(lavin, vis_processor, device=torch.device('cuda'))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================


def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Type and press Enter',
                                                                    interactive=True), gr.update(value="Upload & Start Chat",
                                                                                                 interactive=True), chat_state, img_list


def upload_img(gr_img, text_input, chat_state):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(gr_img, chat_state, img_list)
    return gr.update(interactive=False), gr.update(interactive=True,
                                                   placeholder='Type and press Enter'), gr.update(value="Start Chatting",
                                                                                                  interactive=False), chat_state, img_list


def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    if chat_state is None:
        chat_state = CONV_VISION.copy()
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, temperature, top_p, sampling_seed):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              temperature=temperature,
                              top_p=top_p,
                              sampling_seed=sampling_seed,
                              max_new_tokens=300,
                              max_length=2000)
    chatbot[-1][1] = llm_message
    return chatbot, chat_state, img_list


title = """<h1 align="center">Demo</h1>"""
description = """<h3>Upload your images and start chatting!</h3>"""

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil")
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")

            top_p = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.95,
                step=0.05,
                interactive=True,
                label="top_k in sampling",
            )

            sampling_seed = gr.Slider(
                minimum=0,
                maximum=100,
                value=23,
                step=1,
                interactive=True,
                label="sampling_seed",
            )

            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='System')
            prompt = "Describe the image in the aspects of disaster type, cause, detailed observations, and future risk using the following template. Template: { Type : [TXT], Cause : [TXT], Observation : [[TXT], [TXT], [TXT], ...], Future risk : [TXT] }."
            text_input = gr.Textbox(label='User', placeholder='Type and press Enter', interactive=True, value=prompt)

    upload_button.click(upload_img, [image, text_input, chat_state], [image, text_input, upload_button, chat_state, img_list])

    text_input.submit(gradio_ask, [text_input, chatbot, chat_state],
                      [text_input, chatbot, chat_state]).then(gradio_answer, [chatbot, chat_state, img_list, temperature, top_p, sampling_seed],
                                                              [chatbot, chat_state, img_list])
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, image, text_input, upload_button, chat_state, img_list], queue=False)

demo.launch(share=True, enable_queue=True, server_name="127.0.0.1")
