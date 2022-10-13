#! /usr/bin/env bash
set -euo pipefail

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

docker run -it --entrypoint bash --gpus all --ipc="host" \
	-v "$(pwd)"/../scripts/run_eval.sh:/workspace/fairseq/run_eval.sh \
	-v "$(pwd)"/../scripts/run_inference.sh:/workspace/fairseq/run_inference.sh \
	-v "$(pwd)"/data:/mnt \
	fairseq:latest
