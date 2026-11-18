# Reproducing the π0.5 LoRA Experiment

This repository contains the code modifications, evaluation scripts,
episode splits, normalization statistics, and experiment results for
single-task π0.5 LoRA fine-tuning on LIBERO.

## 1. Clone the original openpi repository

    git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
    cd openpi

Checkout the exact upstream commit listed in
reproducibility/openpi_commit.txt:

    git checkout 215abfb217dbac7d5f1273282331b9b1866c0479

## 2. Apply the experiment modifications

From the pi05-libero-lora-study repository:

    bash scripts/install_into_openpi.sh /path/to/openpi

Install the training and LIBERO evaluation dependencies according to
the upstream openpi instructions.

## 3. Training configuration

- Model: pi05_base
- Method: LoRA + selected non-LLM trainable parameters
- Trainable parameters: 466,957,072 (13.72%)
- Dataset: physical-intelligence/libero
- Target task: pick up the book and place it in the back compartment of the caddy
- Training episodes: 35
- Validation episodes: 6
- Action horizon: 10
- Batch size: 1
- Training steps: 6000
- Warmup steps: 200
- Peak learning rate: 2.5e-5
- GPU: NVIDIA RTX 4090

## 4. Train

Run the following commands inside openpi:

    export OPENPI_EPISODES_FILE="$PWD/results/book_task09/train_episodes.json"

    CUDA_VISIBLE_DEVICES=0 \
    PYTHONFAULTHANDLER=1 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
    ./.venv/bin/python -u -c 'import openpi.training.checkpoints; import torch; import runpy; runpy.run_path("scripts/train.py", run_name="__main__")' \
    pi05_libero_book_lora --exp-name book_6k_v1

The training-only normalization statistics are included in this
repository and restored by install_into_openpi.sh.

## 5. Evaluate

Serve the trained policy from openpi:

    CUDA_VISIBLE_DEVICES=0 \
    ./.venv/bin/python -u -c 'import openpi.training.checkpoints; import torch; import runpy; runpy.run_path("scripts/serve_policy.py", run_name="__main__")' \
    policy:checkpoint \
    --policy.config pi05_libero_book_lora \
    --policy.dir checkpoints/pi05_libero_book_lora/book_6k_v1/5999

Run LIBERO evaluation using examples/libero/eval_book.py in the
separate LIBERO Python environment.

Use 50 trials, seed 7, and replan_steps 5.

## 6. Results

Checkpoint step | Validation Flow Matching Loss | LIBERO Success
2000            | 0.040666                      | 12% (6/50)
4000            | 0.039259                      | 44% (22/50)
5999            | 0.038488                      | 58% (29/50)

The baseline pi05_libero checkpoint achieved 93.8% across all
LIBERO-10 tasks (469/500). This is a different evaluation scope
from the single-task LoRA experiment.

## 7. Notes

The repository does not include model checkpoints, full datasets,
virtual environments, or large evaluation videos.

See README.md for detailed results and failure analysis.
