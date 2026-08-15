#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/chamfer3D/setup.py" install

pushd "${SCRIPT_DIR}/emd_module" >/dev/null
python setup.py install
popd >/dev/null

cmake -S "${SCRIPT_DIR}/evaluate_code" -B "${SCRIPT_DIR}/build/evaluate_code"
cmake --build "${SCRIPT_DIR}/build/evaluate_code" --parallel
cp "${SCRIPT_DIR}/build/evaluate_code/evaluate" "${SCRIPT_DIR}/evaluate_code/evaluate"
