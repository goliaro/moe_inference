# FasterTransformer MoE

FasterTransformer is developed by NVIDIA to highly optimize the encoder and decoder components. It is built on top of CUDA, cuBLAS, cuBLASLt and C++, with API support for the following frameworks: TensorFlow, PyTorch and Triton backend. 

[Source code](https://github.com/NVIDIA/FasterTransformer) with sample tutorial

## NVIDIA Triton Inference Server
[Triton](https://developer.nvidia.com/nvidia-triton-inference-server) is an open-sourced inference server that has a backend integrated with FasterTransformer.  


## Parallelism in FasterTransformer

TODO: figure out what type of parallelism is used by FasterTransformer: data, model, pipeline, tensor parallelism?

FasterTransformer supports tensor, pipeline and model parallelism.

Example of parallelism in [GPT model with FasterTransformer](https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md):
- Tensor parallelism
  - follows the concept in [Megatron](https://arxiv.org/pdf/1909.08053.pdf) (Pg 4)
  - For both self-attention block and feed forward network block, split the weights of first matrix multiplication by row and split the weights of the second matrix multiplication by column.
  - only requires 2 reduction operation for each transformer block
- Pipeline parallelism
  - splits the whole batch of request into multiple micro batches and hides the bubble of communication
  - adjusts the micro batch size automatically for different cases

Model parallelism:
To prevent the additional work of splitting the model for model parallelism, FasterTransformer also provides a tool to split and convert models from different formats to the FasterTransformer binary file format. Then FasterTransformer can load the model in a binary format directly. 

## Optimizations
- Layer fusion
  - combines multiple layers of NNs into a single one that would be computed with one single kernel. 
  - reduces data transfer and increases math density, thus accelerating computation
  - Eg. all the operations in the multi-head attention block can be combined into one kernel.
- Cache results to avoid recomputation
  - allocates buffer to store previous keys and values 
  - saves the cost of recomputing, allocating a buffer at each step, and the cost of concatenation
  - Eg. caching (Q * K^T) * V computation
- Memory optimization
  - reuses the memory buffer of activations/outputs in different decoder layers
- Usage of MPI and NCCL to enable inter/intra-node communication and support model parallelism
- MatMul kernel autotuning (GEMM autotuning)
  - uses functionalities from CuBLAS and CuTLASS libraries
  - choose different low-level algorithms using the "cublasGemmAlgo_t" input parameter

<!-- ## Synchronization/communication collectives

TODO: figure out what synchronization collectives are used. Eg: SPMD, gang-scheduling, MPMD, etc...?

(Not sure if this is what we're looking for, but it seems to be something useful to consider if we're designing our own all-to-all)

FasterTransformer uses NCCL (NVIDIA Collective Communication Library)
NCCL implements GPU-accelerates collective operations:
- all-gather
- all-reduce
- broadcast
- reduce
- reduce-scatter
- point-to-point send and receive

NCCL is topology-aware and the new [PXN](https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/) feature is introduced to deliver full bandwidth for a GPU to communicate with a NIC.
- the communication goes through NVLink and then PCI instead of going through the CPU using QPI or other inter-CPU protocols
- the GPU still tries to use its local NIC as much as possible, but it can reach other NICs if required. -->

<!-- ## Implementing a MoE model in FasterTransformer

TODO: add support for MoE by writing our own implementation

(I'm kind of confused on which of these starting guides to look at and how to set it up for MoE specifically, will find a time this week to meet with Gabriele, thanks in advance!)

[Deploying GPT-J and T5 with NVIDIA Triton Inference Server](https://developer.nvidia.com/blog/deploying-gpt-j-and-t5-with-fastertransformer-and-triton-inference-server/)

[FasterTransformer docs](https://github.com/NVIDIA/FasterTransformer/tree/main/docs) -->

## Pretrained MoE model

We are using the modelscope MoE model from [here](https://www.modelscope.cn/models/PAI/nlp_gpt3_text-generation_0.35B_MoE-64/summary). The model consists of the following files:

<details>
<summary><b>Original model files</b></summary>
<br>

```bash
layer_0_expert_0_mp_rank_00_model_states.pt
...
layer_0_expert_63_mp_rank_00_model_states.pt
layer_1_expert_0_mp_rank_00_model_states.pt
...
layer_1_expert_63_mp_rank_00_model_states.pt
...
layer_2_expert_63_mp_rank_00_model_states.pt
layer_3_expert_0_mp_rank_00_model_states.pt
...
layer_3_expert_63_mp_rank_00_model_states.pt
layer_4_expert_0_mp_rank_00_model_states.pt
...
layer_4_expert_63_mp_rank_00_model_states.pt
layer_5_expert_0_mp_rank_00_model_states.pt
...
layer_5_expert_63_mp_rank_00_model_states.pt
layer_6_expert_0_mp_rank_00_model_states.pt
...
layer_6_expert_63_mp_rank_00_model_states.pt
layer_7_expert_0_mp_rank_00_model_states.pt
...
layer_7_expert_63_mp_rank_00_model_states.pt
layer_8_expert_0_mp_rank_00_model_states.pt
...
layer_8_expert_63_mp_rank_00_model_states.pt
layer_9_expert_0_mp_rank_00_model_states.pt
...
layer_9_expert_63_mp_rank_00_model_states.pt
layer_10_expert_0_mp_rank_00_model_states.pt
...
layer_10_expert_63_mp_rank_00_model_states.pt
layer_11_expert_0_mp_rank_00_model_states.pt
...
layer_11_expert_63_mp_rank_00_model_states.pt
mp_rank_00
```

</details>

In other words, we have 64 expert files for the 12 layers 0-11. We then have the `mp_rank_00` file, presumably containing all data for the remaining layers.

### Converting model to FasterTransformer format

When converting the Modelscope model to FasterTransformer format, we obtain a `c-models/<N>-gpu` folder with the following model files:

<details>
<summary><b>1-GPU converted FasterTransformer checkpoint</b></summary>
<br>

```bash
model.wpe.bin
model.wte.bin

model.final_layernorm.bias.bin
model.final_layernorm.weight.bin

model.layers.0.attention.dense.bias.bin
model.layers.0.attention.dense.weight.0.bin
model.layers.0.attention.query_key_value.bias.0.bin
model.layers.0.attention.query_key_value.weight.0.bin
model.layers.0.input_layernorm.bias.bin
model.layers.0.input_layernorm.weight.bin
    model.layers.0.mlp.dense_4h_to_h.bias.bin
    model.layers.0.mlp.dense_4h_to_h.weight.0.bin
    model.layers.0.mlp.dense_h_to_4h.bias.0.bin
    model.layers.0.mlp.dense_h_to_4h.weight.0.bin
model.layers.0.post_attention_layernorm.bias.bin
model.layers.0.post_attention_layernorm.weight.bin

model.layers.1.attention.dense.bias.bin
    model.layers.1.mlp.moe.gate.wg.weight.bin
model.layers.1.attention.dense.weight.0.bin
model.layers.1.attention.query_key_value.bias.0.bin
model.layers.1.attention.query_key_value.weight.0.bin
model.layers.1.input_layernorm.bias.bin
model.layers.1.input_layernorm.weight.bin
    model.layers.1.mlp.moe.experts.dense_4h_to_h.bias.bin
    model.layers.1.mlp.moe.experts.dense_4h_to_h.weight.0.bin
    model.layers.1.mlp.moe.experts.dense_h_to_4h.bias.0.bin
    model.layers.1.mlp.moe.experts.dense_h_to_4h.weight.0.bin
model.layers.1.post_attention_layernorm.bias.bin
model.layers.1.post_attention_layernorm.weight.bin

...
...

model.layers.23.attention.dense.bias.bin
    model.layers.23.mlp.moe.gate.wg.weight.bin
model.layers.23.attention.dense.weight.0.bin
model.layers.23.attention.query_key_value.bias.0.bin
model.layers.23.attention.query_key_value.weight.0.bin
model.layers.23.input_layernorm.bias.bin
model.layers.23.input_layernorm.weight.bin
    model.layers.23.mlp.moe.experts.dense_4h_to_h.bias.bin
    model.layers.23.mlp.moe.experts.dense_4h_to_h.weight.0.bin
    model.layers.23.mlp.moe.experts.dense_h_to_4h.bias.0.bin
    model.layers.23.mlp.moe.experts.dense_h_to_4h.weight.0.bin
model.layers.23.post_attention_layernorm.bias.bin
model.layers.23.post_attention_layernorm.weight.bin
```

</details>

You can see how every other layer (odd-numbered layers) has a MoE component, and every other one (even-numbered layers) does not. In total, we have 12 MoE layer and 12 non-MoE layers.

If we use 4 GPUs instead of 1, some of the files will be split into 4 parts, as follows (see indented files):

<details>
<summary><b>1-GPU converted FasterTransformer checkpoint</b></summary>
<br>

```bash
model.wpe.bin
model.wte.bin

model.final_layernorm.bias.bin
model.final_layernorm.weight.bin

model.layers.0.attention.dense.bias.bin
    model.layers.0.attention.dense.weight.0.bin
    model.layers.0.attention.dense.weight.1.bin
    model.layers.0.attention.dense.weight.2.bin
    model.layers.0.attention.dense.weight.3.bin
    model.layers.0.attention.query_key_value.bias.0.bin
    model.layers.0.attention.query_key_value.bias.1.bin
    model.layers.0.attention.query_key_value.bias.2.bin
    model.layers.0.attention.query_key_value.bias.3.bin
    model.layers.0.attention.query_key_value.weight.0.bin
    model.layers.0.attention.query_key_value.weight.1.bin
    model.layers.0.attention.query_key_value.weight.2.bin
    model.layers.0.attention.query_key_value.weight.3.bin
model.layers.0.input_layernorm.bias.bin
model.layers.0.input_layernorm.weight.bin
model.layers.0.mlp.dense_4h_to_h.bias.bin
    model.layers.0.mlp.dense_4h_to_h.weight.0.bin
    model.layers.0.mlp.dense_4h_to_h.weight.1.bin
    model.layers.0.mlp.dense_4h_to_h.weight.2.bin
    model.layers.0.mlp.dense_4h_to_h.weight.3.bin
    model.layers.0.mlp.dense_h_to_4h.bias.0.bin
    model.layers.0.mlp.dense_h_to_4h.bias.1.bin
    model.layers.0.mlp.dense_h_to_4h.bias.2.bin
    model.layers.0.mlp.dense_h_to_4h.bias.3.bin
    model.layers.0.mlp.dense_h_to_4h.weight.0.bin
    model.layers.0.mlp.dense_h_to_4h.weight.1.bin
    model.layers.0.mlp.dense_h_to_4h.weight.2.bin
    model.layers.0.mlp.dense_h_to_4h.weight.3.bin
model.layers.0.post_attention_layernorm.bias.bin
model.layers.0.post_attention_layernorm.weight.bin

...
...

model.layers.23.attention.dense.bias.bin
model.layers.23.mlp.moe.gate.wg.weight.bin
    model.layers.23.attention.dense.weight.0.bin
    model.layers.23.attention.dense.weight.1.bin
    model.layers.23.attention.dense.weight.2.bin
    model.layers.23.attention.dense.weight.3.bin
    model.layers.23.attention.query_key_value.bias.0.bin
    model.layers.23.attention.query_key_value.bias.1.bin
    model.layers.23.attention.query_key_value.bias.2.bin
    model.layers.23.attention.query_key_value.bias.3.bin
    model.layers.23.attention.query_key_value.weight.0.bin
    model.layers.23.attention.query_key_value.weight.1.bin
    model.layers.23.attention.query_key_value.weight.2.bin
    model.layers.23.attention.query_key_value.weight.3.bin
model.layers.23.input_layernorm.bias.bin
model.layers.23.input_layernorm.weight.bin
    model.layers.23.mlp.moe.experts.dense_4h_to_h.bias.bin
    model.layers.23.mlp.moe.experts.dense_4h_to_h.weight.0.bin
    model.layers.23.mlp.moe.experts.dense_4h_to_h.weight.1.bin
    model.layers.23.mlp.moe.experts.dense_4h_to_h.weight.2.bin
    model.layers.23.mlp.moe.experts.dense_4h_to_h.weight.3.bin
    model.layers.23.mlp.moe.experts.dense_h_to_4h.bias.0.bin
    model.layers.23.mlp.moe.experts.dense_h_to_4h.bias.1.bin
    model.layers.23.mlp.moe.experts.dense_h_to_4h.bias.2.bin
    model.layers.23.mlp.moe.experts.dense_h_to_4h.bias.3.bin
    model.layers.23.mlp.moe.experts.dense_h_to_4h.weight.0.bin
    model.layers.23.mlp.moe.experts.dense_h_to_4h.weight.1.bin
    model.layers.23.mlp.moe.experts.dense_h_to_4h.weight.2.bin
    model.layers.23.mlp.moe.experts.dense_h_to_4h.weight.3.bin
model.layers.23.post_attention_layernorm.bias.bin
model.layers.23.post_attention_layernorm.weight.bin
```

</details>

In addition, we get the following files: 

```
args.txt
config.ini
merges.txt
vocab.json
```

The contents of the `args.txt` and `config.ini` config files are below:

<details>
<summary><b>argx.txt file</b></summary>
<br>

```
num_layers:24
hidden_size:1024
ffn_hidden_size:4096
num_attention_heads:16
kv_channels:64
max_position_embeddings:2048
make_vocab_size_divisible_by:128
layernorm_epsilon:1e-05
apply_residual_connection_post_layernorm:False
openai_gelu:False
onnx_safe:None
bert_binary_head:True
num_experts:[64]
attention_dropout:0.1
hidden_dropout:0.1
weight_decay:0.1
start_weight_decay:0.1
end_weight_decay:0.1
weight_decay_incr_style:constant
clip_grad:1.0
adam_beta1:0.9
adam_beta2:0.95
adam_eps:1e-08
sgd_momentum:0.9
micro_batch_size:4
global_batch_size:256
rampup_batch_size:None
recompute_granularity:selective
distribute_saved_activations:False
recompute_method:None
recompute_num_layers:1
train_iters:572204
train_samples:None
log_interval:1
exit_interval:None
exit_duration_in_mins:None
exit_signal_handler:False
tensorboard_dir:/mnt/output_wudao/tensorboard/wudao-megatron-gpt-moe-64-0.35B-lr-3e-4-bs-4-gbs-256-tp-1-ac-sel-zero-none_2022.12.14-01.13.02
masked_softmax_fusion:True
bias_gelu_fusion:True
bias_dropout_fusion:True
optimizer:adam
dataloader_type:single
async_tensor_model_parallel_allreduce:True
no_persist_layer_norm:False
sequence_parallel:False
gradient_accumulation_fusion:False
seed:1234
data_parallel_random_init:False
init_method_std:0.006
init_method_xavier_uniform:False
lr:0.0003
lr_decay_style:linear
lr_decay_iters:572204
lr_decay_samples:None
lr_warmup_fraction:None
lr_warmup_iters:715
lr_warmup_samples:0
min_lr:3e-05
override_opt_param_scheduler:False
use_checkpoint_opt_param_scheduler:False
save:/mnt/output_wudao/checkpoint/wudao-megatron-gpt-moe-64-0.35B-lr-3e-4-bs-4-gbs-256-tp-1-ac-sel-zero-none
save_interval:10000
no_save_optim:None
no_save_rng:None
load:/mnt/output_wudao/checkpoint/wudao-megatron-gpt-moe-64-0.35B-lr-3e-4-bs-4-gbs-256-tp-1-ac-sel-zero-none
no_load_optim:None
no_load_rng:None
finetune:False
perform_initialization:True
use_checkpoint_args:False
fp16:True
bf16:False
loss_scale:None
initial_loss_scale:4294967296
min_loss_scale:1.0
loss_scale_window:1000
hysteresis:2
fp32_residual_connection:False
apply_query_key_layer_scaling:True
attention_softmax_in_fp32:False
accumulate_allreduce_grads_in_fp32:False
fp16_lm_cross_entropy:False
tensor_model_parallel_size:1
pipeline_model_parallel_size:1
pipeline_model_parallel_split_rank:None
num_layers_per_virtual_pipeline_stage:None
distributed_backend:nccl
DDP_impl:local
use_contiguous_buffers_in_local_ddp:True
scatter_gather_tensors_in_pipeline:True
local_rank:0
lazy_mpu_init:None
use_cpu_initialization:None
empty_unused_memory_level:0
standalone_embedding_stage:False
use_distributed_optimizer:False
eval_iters:10
eval_interval:100
data_path:['/mnt/wudao/wudao_jiebabpe_text_document']
split:98,2,0
vocab_file:tokenizer.json
merge_file:None
vocab_extra_ids:0
seq_length:2048
encoder_seq_length:2048
decoder_seq_length:None
retriever_seq_length:256
sample_rate:1.0
mask_prob:0.15
short_seq_prob:0.1
mmap_warmup:False
num_workers:2
tokenizer_type:JiebaBPETokenizer
data_impl:mmap
reset_position_ids:False
reset_attention_mask:False
eod_mask_loss:False
adlr_autoresume:False
adlr_autoresume_interval:1000
ict_head_size:None
biencoder_projection_dim:0
biencoder_shared_query_context_model:False
ict_load:None
bert_load:None
titles_data_path:None
query_in_block_prob:0.1
use_one_sent_docs:False
evidence_data_path:None
retriever_report_topk_accuracies:[]
retriever_score_scaling:False
block_data_path:None
embedding_path:None
indexer_batch_size:128
indexer_log_interval:1000
num_classes:1000
img_h:224
img_w:224
num_channels:3
patch_dim:16
classes_fraction:1.0
data_per_class_fraction:1.0
data_sharding:True
head_lr_mult:1.0
vision_pretraining:False
vision_pretraining_type:classify
vision_backbone_type:vit
swin_backbone_type:tiny
mask_type:random
mask_factor:1.0
iter_per_epoch:1250
dino_local_img_size:96
dino_local_crops_number:10
dino_head_hidden_size:2048
dino_bottleneck_size:256
dino_freeze_last_layer:1
dino_norm_last_layer:False
dino_warmup_teacher_temp:0.04
dino_teacher_temp:0.07
dino_warmup_teacher_temp_epochs:30
log_params_norm:False
log_num_zeros_in_grad:False
tensorboard_log_interval:1
tensorboard_queue_size:1
log_timers_to_tensorboard:True
log_batch_size_to_tensorboard:True
log_learning_rate_to_tensorboard:True
log_loss_scale_to_tensorboard:True
log_validation_ppl_to_tensorboard:True
log_memory_to_tensorboard:False
log_world_size_to_tensorboard:False
inference_batch_times_seqlen_threshold:512
pretrained_model_name_or_path:/mnt/output_wudao/checkpoint/wudao-megatron-gpt-moe-64-0.35B-lr-3e-4-bs-4-gbs-256-tp-1-ac-sel-zero-none
use_expert_residual_network:False
top_k_linear_strategy:standard
profile_flops:False
onnx_runtime_training:False
zero_1_memory_optimization:False
zero_2_memory_optimization:False
zero_3_memory_optimization:False
cpu_offload:False
epochs:None
logging_level:info
logging_path:/mnt/output_wudao/log/wudao-megatron-gpt-moe-64-0.35B-lr-3e-4-bs-4-gbs-256-tp-1-ac-sel-zero-none_2022.12.14-01.13.02
data_dir:None
data_name:None
task:None
app_name:None
user_defined_parameters:None
user_script:None
cfg_path:None
adaptive_seq_len:False
eval_fp32:False
sparse_attention:None
sparse_attention_block_size:16
rotary_position_embeddings:False
valid_data_path:None
eval_tasks:None
num_fewshot:None
expert_interval:2
inference:False
train_tokens:None
mlp_type:standard
split_transformers:False
partition_activations:False
contigious_checkpointing:False
checkpoint_in_cpu:False
synchronize_each_layer:False
profile_backward:False
remote_device:none
reset_iteration:False
topk:1
noisy_gate_policy:None
moe_token_dropping:True
moe_train_capacity_factor:1.0
moe_eval_capacity_factor:1.0
moe_min_capacity:4
moe_loss_coeff:0.01
create_moe_param_group:False
use_tutel:False
load_ds_moe_ckpts:False
open_oss_save:True
enable_expert_tensor_parallelism:False
override_lr_scheduler:True
lr_decay_tokens:None
lr_warmup_tokens:None
use_checkpoint_lr_scheduler:False
poc_type:standard
rank:0
world_size:8
transformer_pipeline_model_parallel_size:1
data_parallel_size:8
virtual_pipeline_model_parallel_size:None
params_dtype:torch.float16
consumed_train_samples:17920000
consumed_valid_samples:1792000
no_pipeline_parallel:True
moe_expert_parallel_size:8
padded_vocab_size:51200
device:0
model_type:ModelType.encoder_or_decoder
iteration:0
do_train:1
do_valid:1
do_test:0

```

</details>

<details>
<summary><b>config.ini file</b></summary>
<br>

```
[gpt]
model_name = gpt
head_num = 16
size_per_head = 64
inter_size = 4096
num_layer = 24
max_pos_seq_len = 2048
vocab_size = 51200
has_adapters = False
adapter_inter_size = 0
layernorm_eps = 1e-05
start_id = 50256
end_id = 50256
weight_data_type = fp16
tensor_para_size = 1

[structure]
gpt_with_moe = 1
expert_num = 64
moe_layers = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]

```

</details>

The `args.txt` file is identical for both the 1-GPU and 4-GPUs versions of the model. The `config.ini` only varies in the `tensor_para_size = 1` or `tensor_para_size = 4` parameter.

## Benchmarking results

TODO
