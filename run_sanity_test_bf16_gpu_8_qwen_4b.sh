#!/bin/bash
set -x

if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME="Qwen3-4B"
fi

if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="/mnt/public/data_science/model/llm/Qwen/Qwen3-4B"
fi

if [ -z "$ALGO" ]; then
    # ALGO=PPO-Token-TIS
    ALGO=PG-Seq-IS
fi

if [ -z "$DTYPE" ]; then
    # DTYPE=float16
    DTYPE=bfloat16
fi

if [ -z "$LOSS_AGG_MODE" ]; then
    LOSS_AGG_MODE=seq-mean-token-sum-norm
fi

if [ -z "$DEVICE_NUM" ]; then
    DEVICE_NUM=8
fi

echo $MODEL_NAME
echo $MODEL_PATH
echo $ALGO
echo $DTYPE
echo $DEVICE_NUM
echo "${@:1}"

# Train over a single node, 8 A100-80GB GPUs.
RAY_DEDUP_LOGS=0 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=./sanity_test/math_1460.parquet \
    data.val_files=[./sanity_test/aime_2024.parquet,./sanity_test/aime_2025.parquet] \
    data.train_batch_size=64 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.policy_loss.algo=$ALGO \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.dtype=$DTYPE \
    actor_rollout_ref.rollout.dtype=$DTYPE \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    trainer.project_name=precision-rl \
    trainer.experiment_name=sanity_test-$DTYPE-$ALGO-$MODEL_NAME-GPU-$DEVICE_NUM \
    trainer.val_before_train=True \
    trainer.total_epochs=20 \
    trainer.n_gpus_per_node=$DEVICE_NUM "${@:1}"
