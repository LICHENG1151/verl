#!/bin/bash
set -x

if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="/mnt/public/data_science/models/downloaded_models/DeepSeek-R1-Distill-Qwen-1.5B"
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

echo $MODEL_PATH
echo $ALGO
echo $DTYPE
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
    trainer.experiment_name=sanity_test-$DTYPE-$ALGO-8-gpu \
    trainer.val_before_train=True \
    trainer.total_epochs=20 \
    trainer.n_gpus_per_node=8 "${@:1}"
