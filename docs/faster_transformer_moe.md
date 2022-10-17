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

## Synchronization/communication collectives

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
- the GPU still tries to use its local NIC as much as possible, but it can reach other NICs if required.

## Implementing a MoE model in FasterTransformer

TODO: add support for MoE by writing our own implementation

(I'm kind of confused on which of these starting guides to look at and how to set it up for MoE specifically, will find a time this week to meet with Gabriele, thanks in advance!)

[Deploying GPT-J and T5 with NVIDIA Triton Inference Server](https://developer.nvidia.com/blog/deploying-gpt-j-and-t5-with-fastertransformer-and-triton-inference-server/)

[FasterTransformer docs](https://github.com/NVIDIA/FasterTransformer/tree/main/docs)

## Pretrained MoE model

TODO: figure out whether we can reuse the [Fairseq pre-trained MoE model](https://github.com/facebookresearch/fairseq/tree/main/examples/moe_lm) or we need to obtain another one

## Benchmarking results

TODO
