#!/bin/bash
set -e
set -x

git clone https://github.com/NVIDIA/FasterTransformer.git
mkdir -p FasterTransformer/build
cd FasterTransformer/build
git submodule init && git submodule update
# export ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader|head -n 1)
# echo "Compiling with NVIDIA GPU arch: $ARCH"
# export ARCH=$(( ARCH * 10  ))
#cmake -DSM=$ARCH -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
make -j

pip install -r ../examples/pytorch/gpt/requirement.txt

wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -P ../models
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -P ../models

pip install git+https://github.com/microsoft/DeepSpeed.git
git clone https://github.com/byshiue/Megatron-DeepSpeed/ -b moe_ft
pip install Megatron-DeepSpeed/
pip install jieba
pip install -r ../examples/pytorch/gpt/requirement.txt
apt-get update -y && apt-get -y install git-lfs
git lfs clone https://www.modelscope.cn/PAI/nlp_gpt3_text-generation_0.35B_MoE-64.git
mv nlp_gpt3_text-generation_0.35B_MoE-64 ../models
PYTHONPATH=$PWD/../ python ../examples/pytorch/gpt/utils/megatron_gpt_moe_ckpt_convert.py \
                    --input-dir ../models/nlp_gpt3_text-generation_0.35B_MoE-64/model \
                    --saved-dir ../models/nlp_gpt3_text-generation_0.35B_MoE-64/model/c-models \
                    --infer-gpu-num 1 \
                    --vocab-path ../models/gpt2-vocab.json \
                    --merges-path ../models/gpt2-merges.txt

echo \
'据悉,自驾
“首金”花落谁家,无疑' > sample_input_file.txt

python3 ../examples/pytorch/gpt/multi_gpu_gpt_example.py \
        --tensor_para_size=1 \
        --pipeline_para_size=1 \
        --ckpt_path=../models/nlp_gpt3_text-generation_0.35B_MoE-64/model/c-models/1-gpu/ \
        --data_type=fp16 \
        --vocab_file=../models/nlp_gpt3_text-generation_0.35B_MoE-64/tokenizer.json \
        --vocab_size=51200 \
        --start_id=7 \
        --end_id=7 \
        --sample_input_file=sample_input_file.txt \
        --use_jieba_tokenizer
