# python3 src/encode.py \
#     --model_name=llama3.2_1b \
#     --dataset=cwq \
#     --sample=300 \
#     --per_device_train_batch_size=1 \
#     --num_train_epochs=1 \
#     --learning_rate=0.0003 \
#     --lora_rank=2 \
#     --lora_alpha=32 

python3 src/inference.py \
    --model_name=qwen2.5_1.5b \
    --dataset=cwq \
    --sample=300 \
    --num_train_epochs=1 \
    --learning_rate=0.0003 \
    --lora_rank=2 \
    --lora_alpha=32 \
    --max_new_tokens=20 \
    --inference_method=prag 