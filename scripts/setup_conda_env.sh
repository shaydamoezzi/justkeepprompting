#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAMBA_DIR="${ROOT}/.local-micromamba"
MAMBA_ROOT_PREFIX="${ROOT}/.micromamba"
MAMBA_BIN="${MAMBA_DIR}/micromamba"
ENV_NAME="jkp-selfhosted"
ENV_FILE="${ROOT}/environment.self_hosted.yml"

mkdir -p "${MAMBA_DIR}"

if [ ! -x "${MAMBA_BIN}" ]; then
  TMP_TAR="$(mktemp)"
  curl -L "https://micro.mamba.pm/api/micromamba/linux-64/latest" -o "${TMP_TAR}"
  tar -xjf "${TMP_TAR}" -C "${MAMBA_DIR}" --strip-components=1 bin/micromamba
  rm -f "${TMP_TAR}"
fi

export MAMBA_ROOT_PREFIX

"${MAMBA_BIN}" create -y -f "${ENV_FILE}"

echo
echo "Environment created."
echo "Activate it with:"
echo "  eval \"\$(${MAMBA_BIN} shell hook -s bash)\""
echo "  micromamba activate ${ENV_NAME}"
