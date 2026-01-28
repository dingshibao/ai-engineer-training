pip install ms-swift -U

# 对Qwen2.5-7B-Instruct进行自我认知微调
# 22GB
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \                #####  1
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \                #####   0.1
    --lora_rank 8 \                       #####   4
    --lora_alpha 32 \      
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author WilsonYin \
    --model_name Robot-J.A.R.V.I.S




# 使用交互式命令行进行推理
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/v0-20250817-154340/checkpoint-94 \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048


# merge-lora并使用vLLM进行推理加速
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/v0-20250817-154340/checkpoint-94 \
    --stream true \
    --merge_lora true \
    --infer_backend vllm \
    --vllm_max_model_len 8192 \
    --temperature 0 \
    --max_new_tokens 2048

# web 界面
swift web-ui