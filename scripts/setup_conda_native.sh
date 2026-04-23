#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_HOME="${HOME}/.miniforge3"
INSTALLER="${ROOT}/.tmp-Miniforge3.sh"
ENV_FILE="${ROOT}/environment.self_hosted.yml"
REQ_FILE="${ROOT}/requirements.self_hosted.txt"
ENV_NAME="jkp-selfhosted"

if [ ! -x "${CONDA_HOME}/bin/conda" ]; then
  curl -L "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" -o "${INSTALLER}"
  bash "${INSTALLER}" -b -p "${CONDA_HOME}"
  rm -f "${INSTALLER}"
fi

source "${CONDA_HOME}/etc/profile.d/conda.sh"

# Use Conda for the base environment and pip for the heavyweight GPU/model stack.
# This is much faster and more reliable than asking Conda to solve the full CUDA/PyTorch graph.
conda config --set solver libmamba >/dev/null 2>&1 || true

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -f "${ENV_FILE}" --prune
else
  conda env create -f "${ENV_FILE}"
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
python -m pip install -r "${REQ_FILE}"

echo
echo "Conda installed at ${CONDA_HOME}"
echo "Activate with:"
echo "  source ${CONDA_HOME}/etc/profile.d/conda.sh"
echo "  conda activate ${ENV_NAME}"
