#!/bin/bash
set -e

DATA_PATH=/mnt/data-bin/wikitext-103
MODEL_PATH=/mnt/en_moe_lm_15b/model.pt
#export NCCL_DEBUG=INFO
python -m fairseq_cli.generate $DATA_PATH \
  --path $MODEL_PATH \
  --gen-subset valid \
  --sample-break-mode none \
  --tokens-per-sample 2048 \
  --batch-size 1 \
  --fp16 \
  --is-moe \
  --distributed-world-size 4 \
  --max-valid-steps 4 \
  --model-overrides "{'world_size': 4, 'moe_eval_capacity_token_fraction': 0.05}"

#--output-word-probs
#--log-file
