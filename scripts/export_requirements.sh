#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT="${OUTPUT:-requirements.txt}"

temporary_file="$(mktemp)"
trap 'rm -f "${temporary_file}"' EXIT

"${PYTHON_BIN}" -m pip freeze --local \
  | grep -vE '^(-e |graph-transfer-gpp @ file:)' \
  > "${temporary_file}"

torch_version="$(${PYTHON_BIN} -c 'import torch; print(torch.__version__.split("+")[0])')"
cuda_tag="$(${PYTHON_BIN} -c 'import torch; print("cu" + torch.version.cuda.replace(".", ""))')"

{
  echo "--extra-index-url https://download.pytorch.org/whl/${cuda_tag}"
  echo "--find-links https://data.pyg.org/whl/torch-${torch_version}+${cuda_tag}.html"
  echo
  cat "${temporary_file}"
} > "${OUTPUT}"

if [[ ! -s "${OUTPUT}" ]]; then
  echo "Refusing to keep an empty requirements file." >&2
  exit 1
fi

echo "Wrote $(wc -l < "${OUTPUT}") packages to ${OUTPUT}"
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -m pip --version
