#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_DIR}"

mapfile -t DATASET_NAMES < <(
    uv run python -c \
        'import atria_datasets; print(*sorted(atria_datasets.datasets.list()), sep="\n")'
)

FAILED_DATASETS=()

for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    echo
    echo "=== Preparing ${DATASET_NAME} ==="

    if uv run python usage/prepare_dataset.py "${DATASET_NAME}"; then
        echo "=== Passed: ${DATASET_NAME} ==="
    else
        echo "=== Failed: ${DATASET_NAME} ===" >&2
        FAILED_DATASETS+=("${DATASET_NAME}")
    fi
done

echo
echo "Prepared $((${#DATASET_NAMES[@]} - ${#FAILED_DATASETS[@]}))/${#DATASET_NAMES[@]} datasets successfully."

if [[ ${#FAILED_DATASETS[@]} -gt 0 ]]; then
    echo "Failed datasets: ${FAILED_DATASETS[*]}" >&2
    exit 1
fi
