# Fairseq's 15B-parameters LM MoE model

This document contains a walkthrough the 15B-parameters pre-trained MoE language model from Fairseq. The checkpoint is [available here](https://github.com/facebookresearch/fairseq/tree/main/examples/moe_lm). 

## Model architecture
The Fairseq MoE model from the MoE 15B checkpoint has essentially a GPT structure, with 12 `TransformerDecoderLayers` where a MoE is inserted in every other layer. No encoder layer is used. Each MoE has a Top 2 gating function and 512 experts in total. 

The default expert parallelism is such that each process (corresponding to a GPU) will be assigned a number of expert that is equal to `int(checkpoint_files_count / world_size)) * 8` ([see here](https://github.com/gabrieleoliaro/fairseq/blob/79b0f43322c97e38a7d1de346b97edac135766f8/fairseq/moe_checkpoint_utils.py#L141)), where `checkpoint_files_count=64` and `world_size` is the number of processes/GPUs made available to Fairseq. In the original setup, `world_size=8`, so each process/GPU was assigned `int(64/8)*8=64 experts`, for a total of `64*8=512` experts. In our case, we used a `world_size=4`, so each process/GPU was assigned `int(64/4)*8=128 experts` (see below), for a total of `128*4=512` experts.

<details>
<summary><b>Full structure of the model</b></summary>
<br>

```
TransformerLanguageModel(
  (decoder): TransformerDecoder(
    (dropout_module): FairseqDropout(p=0.1)
    (embed_tokens): Embedding(50264, 768, padding_idx=1)
    (embed_positions): SinusoidalPositionalEmbedding()
    (layers): ModuleList(
      (0): TransformerDecoderLayer(
        [checkpointed] 
        (dropout_module): FairseqDropout(p=0.1)
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout(p=0.1)
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (activation_dropout_module): FairseqDropout(p=0.0)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (1): TransformerDecoderLayer(
        [checkpointed] 
        (dropout_module): FairseqDropout(p=0.1)
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout(p=0.1)
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (moe_layer): MOELayer()
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (2): TransformerDecoderLayer(
        [checkpointed] 
        (dropout_module): FairseqDropout(p=0.1)
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout(p=0.1)
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (activation_dropout_module): FairseqDropout(p=0.0)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (3): TransformerDecoderLayer(
        [checkpointed] 
        (dropout_module): FairseqDropout(p=0.1)
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout(p=0.1)
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (moe_layer): MOELayer()
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (4): TransformerDecoderLayer(
        [checkpointed] 
        (dropout_module): FairseqDropout(p=0.1)
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout(p=0.1)
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (activation_dropout_module): FairseqDropout(p=0.0)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (5): TransformerDecoderLayer(
        [checkpointed] 
        (dropout_module): FairseqDropout(p=0.1)
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout(p=0.1)
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (moe_layer): MOELayer()
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (6): TransformerDecoderLayer(
        [checkpointed] 
        (dropout_module): FairseqDropout(p=0.1)
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout(p=0.1)
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (activation_dropout_module): FairseqDropout(p=0.0)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (7): TransformerDecoderLayer(
        [checkpointed] 
        (dropout_module): FairseqDropout(p=0.1)
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout(p=0.1)
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (moe_layer): MOELayer()
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (8): TransformerDecoderLayer(
        [checkpointed] 
        (dropout_module): FairseqDropout(p=0.1)
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout(p=0.1)
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (activation_dropout_module): FairseqDropout(p=0.0)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (9): TransformerDecoderLayer(
        [checkpointed] 
        (dropout_module): FairseqDropout(p=0.1)
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout(p=0.1)
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (moe_layer): MOELayer()
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (10): TransformerDecoderLayer(
        [checkpointed] 
        (dropout_module): FairseqDropout(p=0.1)
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout(p=0.1)
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (activation_dropout_module): FairseqDropout(p=0.0)
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (11): TransformerDecoderLayer(
        [checkpointed] 
        (dropout_module): FairseqDropout(p=0.1)
        (self_attn): MultiheadAttention(
          (dropout_module): FairseqDropout(p=0.1)
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (moe_layer): MOELayer()
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
    )
    (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (output_projection): Linear(in_features=768, out_features=50264, bias=False)
  )
)
```

</details>

Each MoE layer, whose details have been omitted in the printout above for the sake of conciseness, has the following structure:

<details>
<summary><b>MoE layer</b></summary>
<br>

```
(moe_layer): MOELayer(
	(gate): Top2Gate(
		(wg): Linear(in_features=768, out_features=512, bias=False)
	)
	(experts): ModuleList(
		(0): FeedForwardNetwork(
			(activation_dropout_module): FairseqDropout(p=0.0)
			(fc1): Linear(in_features=768, out_features=3072, bias=True)
			(fc2): Linear(in_features=3072, out_features=768, bias=True)
			(dropout_module): FairseqDropout(p=0.1)
		)
		(1): FeedForwardNetwork(
			(activation_dropout_module): FairseqDropout(p=0.0)
			(fc1): Linear(in_features=768, out_features=3072, bias=True)
			(fc2): Linear(in_features=3072, out_features=768, bias=True)
			(dropout_module): FairseqDropout(p=0.1)
		)

		...
		...
		...

		(126): FeedForwardNetwork(
			(activation_dropout_module): FairseqDropout(p=0.0)
			(fc1): Linear(in_features=768, out_features=3072, bias=True)
			(fc2): Linear(in_features=3072, out_features=768, bias=True)
			(dropout_module): FairseqDropout(p=0.1)
		)
		(127): FeedForwardNetwork(
			(activation_dropout_module): FairseqDropout(p=0.0)
			(fc1): Linear(in_features=768, out_features=3072, bias=True)
			(fc2): Linear(in_features=3072, out_features=768, bias=True)
			(dropout_module): FairseqDropout(p=0.1)
		)
	)
)

```

</details>


## Checkpoint details
The `en_moe_lm_15b.tar.gz` tarball contains 65 files: `model-shared.pt` and 64 files of the form `model-rank-N.pt`, with `N={0,...,63}`.

A jsonified version of all the files is [available here]() for inspection. All large tensors have been removed using [this script](../utils/jsonify_pretrained_checkpoint.py) for conciseness. By comparing them, we can see that the contents of all the checkpoint files only differs when it comes to the `model` member of the dictionary. 

In particular, each of the `model-rank-N.pt` files contains the parameters from 8 experts from each of the 6 MoE layers (i.e. each `model-rank-N.pt` contains 6x8=48 experts). When combining all 64 `model-rank-N.pt` files, we have 8x64=512 experts available for each MoE layer, which matches the first dimension (512) of the weight matrix of each gating network.

On ther other hand, the `model-shared.pt` contains the parameter of all the other non-expert layers (including the gating networks of the MoE layers).



