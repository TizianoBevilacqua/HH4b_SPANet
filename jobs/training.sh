#!/usr/bin/env bash
set -euo pipefail

# Default values (optional)
NUM_GPU=0
SEED=0

usage() {
    echo "Usage: $0 -o options_file -n run_name -s seed -g num_gpu -m spanet_main_dir -e spanet_env_dir -H home_dir [-- extra_spanet_args]"
    exit 1
}

# Parse named options
while getopts ":o:n:s:g:m:e:H:" opt; do
  case $opt in
    o) OPTIONS_FILE="$OPTARG" ;;
    n) RUN_NAME="$OPTARG" ;;
    s) SEED="$OPTARG" ;;
    g) NUM_GPU="$OPTARG" ;;
    m) SPANET_MAIN_DIR="$OPTARG" ;;
    e) SPANET_ENV_DIR="$OPTARG" ;;
    H) HOME_DIR="$OPTARG" ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
    :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
  esac
done

# Remove parsed options
shift $((OPTIND -1))

# Everything after `--` is forwarded automatically
EXTRA_ARGS=("$@")

# Basic validation
: "${OPTIONS_FILE:?Missing -o}"
: "${RUN_NAME:?Missing -n}"
: "${SPANET_MAIN_DIR:?Missing -m}"
: "${SPANET_ENV_DIR:?Missing -e}"
: "${HOME_DIR:?Missing -H}"

export SEED
export NUM_GPU
export SPANET_MAIN_DIR
export SPANET_ENV_DIR
export HOME="$HOME_DIR"

SPANET_DIR="${SPANET_MAIN_DIR}/SPANet"
HH4b_SPANET_DIR="${SPANET_MAIN_DIR}/HH4b_SPANet"

echo "SPANET_MAIN_DIR: $SPANET_MAIN_DIR"
echo "SPANET_ENV_DIR:  $SPANET_ENV_DIR"
echo "HOME:            $HOME"
echo "NUM_GPU:         $NUM_GPU"
echo "SEED:            $SEED"

# Activate environment
source "${SPANET_ENV_DIR}/bin/activate"

echo "Python version: $(python --version)"
echo "Python path:    $(which python)"

cd "${HH4b_SPANET_DIR}"
echo "Working dir:    $(pwd)"

echo "Extra args passed to spanet.train: ${EXTRA_ARGS[*]}"

# Launch training
python -m spanet.train \
    --options_file "$OPTIONS_FILE" \
    -n "$RUN_NAME" \
    --log_dir "$RUN_NAME" \
    --time_limit 07:00:00:00 \
    --gpus "$NUM_GPU" \
    "${EXTRA_ARGS[@]}"
