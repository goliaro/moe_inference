# Fairseq MoE

The Fairseq MoE model created when loading the MoE 15B checkpoint has essentially a GPT-2 architecture, with 12 `TransformerDecoderLayers` where a MoE is inserted in every other layer. No encoder layer is used. Each MoE has a Top 2 gating function and 128 experts. 

Here is the structure of the model:

<details>
<summary>Toggle here </summary>
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
<summary>Toggle here</summary>
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

