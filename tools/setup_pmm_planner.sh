#!/usr/bin/env bash
set -euo pipefail

# Ensure we are inside a Pixi environment
if [ -z "${PIXI_PROJECT_ROOT:-}" ]; then
    echo "[Setup PMM Planner] Not running inside a Pixi environment; skipping."
    exit 0
fi

if [ ! -f "${PIXI_PROJECT_ROOT}/pixi.lock" ]; then
    echo "[Setup PMM Planner] ERROR: Pixi environment not set up."
    exit 1
fi

PMM_DIR="${PIXI_PROJECT_ROOT}/pmm_uav_planner"

# Clone if missing
if [ ! -d "${PMM_DIR}/.git" ]; then
    echo "[Setup PMM Planner] Cloning repo..."
    git clone https://github.com/rducrist/pmm_uav_planner.git "${PMM_DIR}"
fi

# Check pip availability
if ! command -v pip >/dev/null 2>&1; then
    echo "[Setup PMM Planner] ERROR: pip not installed."
    exit 1
fi

echo "[Setup PMM Planner] Running first pip install -e ."
(
    cd "${PMM_DIR}"
    pip install -e .
)

# Build using the exact required sequence
if [ ! -d "${PMM_DIR}/build" ]; then
    echo "[Setup PMM Planner] Building C++ module via CMake..."

    (
        cd "${PMM_DIR}"
        mkdir build
        cd build

        cmake -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" ..
        make -j"$(nproc)"
    )
fi

echo "[Setup PMM Planner] Running final pip install -e ."
(
    cd "${PMM_DIR}"
    pip install -e .
)

echo "[Setup PMM Planner] PMM Planner is ready!"
