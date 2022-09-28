#!/bin/bash
set -e

dir_path=$(cd $(dirname "${BASH_SOURCE:-$0}") && pwd)

docker run -it --entrypoint bash --gpus all --ipc="host" \
	-v $dir_path/scripts/run_inference.sh:/workspace/fairseq/run_inference.sh \
	-v $dir_path/data:/mnt \
	fairseq_exp:latest
