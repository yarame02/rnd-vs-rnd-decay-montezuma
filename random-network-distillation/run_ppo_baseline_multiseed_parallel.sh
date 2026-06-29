#!/bin/bash

export RCALL_NUM_GPU=1
BASE="/workspace/RND_FINAL_REPORT_EXPORT/rnd_multiseed/ppo_baseline"

for seed in 0 1 2 3 4
do
    RUN_DIR="${BASE}/seed_${seed}"
    mkdir -p "${RUN_DIR}"

    (
        timeout 21000 python run_atari.py \
            --env MontezumaRevengeNoFrameskip-v4 \
            --seed ${seed} \
            --num_env 32 \
            --num-timesteps 15000000 \
            --int_coeff 0 \
            --log_dir "${RUN_DIR}" \
            > "${RUN_DIR}/train.log" 2>&1

        python extract_final_metrics.py \
            --run_dir "${RUN_DIR}" \
            --method ppo_baseline \
            --seed ${seed}
    ) &
done

wait
echo "All PPO baseline seeds finished."
