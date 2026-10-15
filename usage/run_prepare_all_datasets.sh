#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_DIR}"

mapfile -t DATASET_NAMES < <(
    uv run python -c \
        'import atria_datasets; print(*sorted(atria_datasets.datasets.list()), sep="\n")'
)

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <index|all|list>" >&2
    exit 2
fi

if [[ "$1" == "list" ]]; then
    for index in "${!DATASET_NAMES[@]}"; do
        printf '%2d: %s\n' "${index}" "${DATASET_NAMES[index]}"
    done
    exit 0
fi

if [[ "$1" == "all" ]]; then
    for dataset_name in "${DATASET_NAMES[@]}"; do
        uv run python usage/prepare_dataset.py "${dataset_name}"
    done
    exit 0
fi

if [[ ! "$1" =~ ^[0-9]+$ ]] || (( $1 >= ${#DATASET_NAMES[@]} )); then
    echo "Dataset index must be between 0 and $((${#DATASET_NAMES[@]} - 1)), or use 'all' or 'list'." >&2
    exit 2
fi

uv run python usage/prepare_dataset.py "${DATASET_NAMES[$1]}"
