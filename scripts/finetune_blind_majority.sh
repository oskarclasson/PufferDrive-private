#!/usr/bin/env bash
# Fine-tune k_scaled_0036_1000 with ~90% blind traffic that only sees cars ahead driving away; extra args are Hydra overrides.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

BASELINE_DIR=${BASELINE_DIR:-weights/k_scaled_0036_1000}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-4000000000}
REVERSE_PENALTY_PER_SECOND=${REVERSE_PENALTY_PER_SECOND:-0.0}
OFF_ROUTE_PENALTY=${OFF_ROUTE_PENALTY:-0.0}
BLIND_PROB_MAX=${BLIND_PROB_MAX:-0.95}
BLIND_PROB_MIN=${BLIND_PROB_MIN:-0.85}
RUN_NAME=${RUN_NAME:-blind_majority_$(date +%Y%m%d-%H%M%S)}
DATA_DIR=experiments/${RUN_NAME}

echo "run dir: ${DATA_DIR}"
echo "live view: tensorboard --logdir experiments --port 6006"

puffer train puffer_drive \
    load_model_path=${BASELINE_DIR}/final_model.pt \
    run_name=${RUN_NAME} \
    train.data_dir=${DATA_DIR} \
    tb=true \
    train.learning_rate=0.0001 \
    train.total_timesteps=${TOTAL_TIMESTEPS} \
    train.checkpoint_interval=25 \
    env.map_dir=pufferlib/resources/drive/binaries/carla \
    env.num_maps=8 \
    env.compute_eval_metrics=true \
    env.conditioning_speed_scale=1.5 \
    env.obs_lane_heading_signed=false \
    env.obs_lane_speed_limit=false \
    env.obs_stride_merge_segments=false \
    env.stop_signs_enabled=false \
    env.reward_overspeed=0.05 \
    env.reward_reverse_fixed_per_second=${REVERSE_PENALTY_PER_SECOND} \
    env.reward_off_route=${OFF_ROUTE_PENALTY} \
    env.scenario_length=2500 \
    env.resample_frequency=256000 \
    env.partner_blindness_prob=${BLIND_PROB_MAX} \
    env.partner_blindness_prob_min=${BLIND_PROB_MIN} \
    env.partner_blindness_trigger_prob=1.0 \
    env.partner_blindness_corridor=false \
    env.partner_blindness_facing_away=true \
    env.partner_blindness_ghosting=true \
    "$@"
