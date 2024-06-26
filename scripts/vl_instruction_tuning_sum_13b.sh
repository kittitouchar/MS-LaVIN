torchrun --nproc_per_node 4 --master_port 12345 train.py \
    --llm_model 13B\
    --llama_model_path ../data/weights/ \
    --data_path ../data/inst_all.json \
    --max_seq_len 512 \
    --batch_size 2 \
    --accum_iter 1 \
    --epochs 50 \
    --warmup_epochs 0.2 \
    --blr 9e-3 \
    --weight_decay 0.025 \
    --output_dir ./LaVIN-13B-ALL-VLIT/\
    --adapter_type attn\
    --adapter_dim 8\
    --adapter_scale 1\
    --n_prompt 6 \
    --prompt_format QCM-ALE \
    --temperature 5.\
    --visual_adapter_type router \
    --do_pretrain \
    --caption_file ../data/inst_all.json \
    --adapter_path ./LaVIN-13B-VLIT/15-eph-pretrain.pth \
    --do_sum