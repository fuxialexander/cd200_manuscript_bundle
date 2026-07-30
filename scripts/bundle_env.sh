#!/bin/bash
set -euo pipefail

CD200_BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${CD200_BUNDLE_ROOT}"

# Optional Caesarion IO package (sibling checkout or CAESAR_SRC override).
if [ -z "${CAESAR_SRC:-}" ]; then
    if [ -d "${CD200_BUNDLE_ROOT}/../Caesarion/src" ]; then
        CAESAR_SRC="$(cd "${CD200_BUNDLE_ROOT}/../Caesarion/src" && pwd)"
    fi
fi

_prepend_pythonpath() {
    local entry="$1"
    case ":${PYTHONPATH:-}:" in
        *":${entry}:"*) ;;
        *)
            if [ -n "${PYTHONPATH:-}" ]; then
                PYTHONPATH="${entry}:${PYTHONPATH}"
            else
                PYTHONPATH="${entry}"
            fi
            ;;
    esac
}

# Prepend Caesarion first, then bundle src, so local packages win.
if [ -n "${CAESAR_SRC:-}" ]; then
    _prepend_pythonpath "${CAESAR_SRC}"
fi
_prepend_pythonpath "${CD200_BUNDLE_ROOT}/src"
export PYTHONPATH

# Prefer a bundle-local uv env when present; otherwise reuse CD200_UV_ENVIRONMENT
# or a sibling motif_chip_model/.venv if available.
if [ -z "${UV_PROJECT_ENVIRONMENT:-}" ]; then
    if [ -n "${CD200_UV_ENVIRONMENT:-}" ]; then
        UV_PROJECT_ENVIRONMENT="${CD200_UV_ENVIRONMENT}"
    elif [ -d "${CD200_BUNDLE_ROOT}/.venv" ]; then
        UV_PROJECT_ENVIRONMENT="${CD200_BUNDLE_ROOT}/.venv"
    elif [ -d "${CD200_BUNDLE_ROOT}/../motif_chip_model/.venv" ]; then
        UV_PROJECT_ENVIRONMENT="$(cd "${CD200_BUNDLE_ROOT}/../motif_chip_model/.venv" && pwd)"
    else
        UV_PROJECT_ENVIRONMENT="${CD200_BUNDLE_ROOT}/.venv"
    fi
fi
export UV_PROJECT_ENVIRONMENT
export UV_CACHE_DIR="${UV_CACHE_DIR:-${CD200_BUNDLE_ROOT}/.uv-cache}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${CD200_BUNDLE_ROOT}/.numba-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${CD200_BUNDLE_ROOT}/.matplotlib-cache}"

mkdir -p "${UV_CACHE_DIR}" "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

PYTHON_CMD=(uv run --no-sync python)
