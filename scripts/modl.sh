#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/bundle_env.sh"
exec "${PYTHON_CMD[@]}" scripts/modl.py "$@"
