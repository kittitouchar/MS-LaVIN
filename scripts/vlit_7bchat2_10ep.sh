#!/bin/bash
#$-S /bin/bash
#$-cwd
#$-ac d=none
#$-j y
#$-o $HOME/log/$JOB_ID
#$ -N it_7bchat2-10ep
#$-jc gtn-container_g8.24h

# For internet connection
export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL

source ~/anaconda3/etc/profile.d/conda.sh
conda activate lavin-torch2.1

/home/quang/anaconda3/envs/lavin-torch2.1/bin/torchrun --nproc_per_node 8 --master_port 12345 landslide_train.py \
    --llm_model llama-2-7b-chat \
    --llama_model_path ./data/weights/ \
    --data_root ./data/ \
    --max_seq_len 512 \
    --batch_size 4 \
    --accum_iter 1 \
    --epochs 10 \
    --warmup_epochs 0.2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir ./models/LaVIN-2-7Bchat-VLIT-10ep/ \
    --adapter_type attn \
    --adapter_dim 8 \
    --adapter_scale 1 \
    --n_prompt 6 \
    --prompt_format QCM-ALE \
    --temperature 5.\
    --visual_adapter_type router \
    --do_pretrain \
    --wandb_enable