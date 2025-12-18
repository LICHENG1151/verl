#!/bin/bash
set -x

if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME="mistral-3B-ins"
fi

if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="/mnt/public/data_science/model/llm/Mistral/mistral-3B-instruct"
fi

if [ -z "$TRAIN_DATA_PATH" ]; then
    TRAIN_DATA_PATH="/mnt/lc/Precision-RL/Precision-RL-verl/sanity_test/math_1460.parquet"
fi

if [ -z "$VAL_DATA_PATH" ]; then
    VAL_DATA_PATH="/mnt/lc/Precision-RL/Precision-RL-verl/sanity_test/aime_2024.parquet,/mnt/lc/Precision-RL/Precision-RL-verl/sanity_test/aime_2025.parquet"

fi

if [ -z "$ALGO" ]; then
    # ALGO=PPO-Token-TIS
    ALGO=PG-Seq-IS
fi

if [ -z "$DTYPE" ]; then
    DTYPE=float16
    # DTYPE=bfloat16
fi

if [ -z "$LOSS_AGG_MODE" ]; then
    LOSS_AGG_MODE=seq-mean-token-sum-norm
fi

if [ -z "$DEVICE_NUM" ]; then
    DEVICE_NUM=2
fi

echo "MODEL_NAME       = $MODEL_NAME"
echo "MODEL_PATH       = $MODEL_PATH"
echo "TRAIN_DATA_PATH  = $TRAIN_DATA_PATH"
echo "VAL_DATA_PATH    = $VAL_DATA_PATH"
echo "ALGO             = $ALGO"
echo "DTYPE            = $DTYPE"
echo "LOSS_AGG_MODE    = $LOSS_AGG_MODE"
echo "DEVICE_NUM       = $DEVICE_NUM"
echo "Extra args       = ${*@Q}"

# Train over a single node, 2 A100-80GB GPUs.
RAY_DEDUP_LOGS=0 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="$TRAIN_DATA_PATH" \
    data.val_files="[$VAL_DATA_PATH]" \
    data.train_batch_size=64 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.policy_loss.algo=$ALGO \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.dtype=$DTYPE \
    actor_rollout_ref.rollout.dtype=$DTYPE \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    trainer.project_name=precision-rl \
    trainer.experiment_name=sanity_test-$DTYPE-$ALGO-$MODEL_NAME-GPU-$DEVICE_NUM \
    trainer.val_before_train=True \
    trainer.total_epochs=20 \
    trainer.n_gpus_per_node=$DEVICE_NUM "${@:1}"
