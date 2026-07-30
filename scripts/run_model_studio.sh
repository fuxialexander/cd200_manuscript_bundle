#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${BUNDLE_ROOT}"
source scripts/bundle_env.sh
exec "${PYTHON_CMD[@]}" -m model_studio.server "$@"
