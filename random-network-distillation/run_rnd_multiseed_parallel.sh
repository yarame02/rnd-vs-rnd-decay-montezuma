#!/bin/bash

export RCALL_NUM_GPU=1
BASE="/workspace/RND_FINAL_REPORT_EXPORT/rnd_multiseed/rnd_classic"

for seed in 0 1 2 3 4
do
    RUN_DIR="${BASE}/seed_${seed}"
    mkdir -p "${RUN_DIR}"

    (
        timeout 25000 python run_atari.py \
            --env MontezumaRevengeNoFrameskip-v4 \
            --seed ${seed} \
            --num_env 32 \
            --num-timesteps 15000000 \
            --log_dir "${RUN_DIR}" \
            > "${RUN_DIR}/train.log" 2>&1

        python extract_final_metrics.py \
            --run_dir "${RUN_DIR}" \
            --method rnd_classic \
            --seed ${seed}
    ) &
done

wait
echo "All RND classic seeds finished."
