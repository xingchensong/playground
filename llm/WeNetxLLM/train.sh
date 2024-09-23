export HF_HOME=/bucket/output/jfs-hdfs/user/xingchen.song/share/huggingface
torchrun --nproc_per_node=8 --nnodes=1 \
    --rdzv_id=2024 --rdzv_backend="c10d" --rdzv_endpoint="localhost:0" \
    train_qwen.py \
        --device "cuda" \
        --output_dir "exp" \
        --batch_size 8 \
        --num_workers 4 \
        --prefetch 5