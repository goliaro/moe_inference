# Fairseq MoE

## Parallelism

Fairseq by default uses data parallelism only; model parallelism is supported by passing a `model_parallelism_size` parameter greater than 1. Model parallel is implemented using Megatron, in particular, the framework loads Megatron as a submodule using [this fork](https://github.com/ngoyal2707/Megatron-LM) (branch `fairseq`).

## Model architecture 
The Fairseq MoE model created when loading the MoE 15B checkpoint has essentially a GPT structure, with 12 `TransformerDecoderLayers` where a MoE is inserted in every other layer. No encoder layer is used. Each MoE has a Top 2 gating function and 128 experts. 

For the language modeling task (default task for the eval_lm script), the model configs (which determine the model type/structure) are as follows:

<details>
<summary>Toggle here </summary>
<br>

```
{'_name': 'transformer_lm_gpt', 'activation_fn': 'gelu', 'dropout': 0.1, 'attention_dropout': 0.1, 'activation_dropout': 0.0, 'relu_dropout': 0.0, 'decoder_embed_dim': 768, 'decoder_output_dim': 768, 'decoder_input_dim': 768, 'decoder_ffn_embed_dim': 3072, 'decoder_layers': 12, 'decoder_attention_heads': 12, 'decoder_normalize_before': True, 'no_decoder_final_norm': False, 'adaptive_softmax_cutoff': None, 'adaptive_softmax_dropout': 0.0, 'adaptive_softmax_factor': 4.0, 'no_token_positional_embeddings': False, 'share_decoder_input_output_embed': True, 'character_embeddings': False, 'character_filters': '[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]', 'character_embedding_dim': 4, 'char_embedder_highway_layers': 2, 'adaptive_input': False, 'adaptive_input_factor': 4.0, 'adaptive_input_cutoff': None, 'tie_adaptive_weights': False, 'tie_adaptive_proj': False, 'decoder_learned_pos': False, 'layernorm_embedding': False, 'no_scale_embedding': False, 'checkpoint_activations': True, 'offload_activations': False, 'decoder_layerdrop': 0.0, 'decoder_layers_to_keep': None, 'quant_noise_pq': 0.0, 'quant_noise_pq_block_size': 8, 'quant_noise_scalar': 0.0, 'min_params_to_wrap': 100000000, 'alternate_decoder_ffn_embed_dim': 0, 'moe_freq': 2, 'moe_expert_count': 512, 'moe_gating_use_fp32': True, 'moe_second_expert_policy': 'all', 'moe_normalize_gate_prob_before_dropping': False, 'moe_expert_ffn_dim': None, 'moe_top1_expert': False, 'moe_eval_capacity_token_fraction': 0.05, 'moe_normalize_expert_grad': 'sqrt_world_size', 'use_moe_pad_mask': False, 'record_a2a_perf_stats': False, 'dummy_a2a': False, 'use_stable_embedding': False, 'base_layers': 0, 'base_sublayers': 1, 'base_shuffle': 0, 'add_bos_token': False, 'tokens_per_sample': 2048, 'max_target_positions': 2048, 'tpu': False, 'memory_efficient_fp16': True, 'fp16': True, 'fp16_no_flatten_grads': False, 'ddp_backend': 'fully_sharded', 'world_size': 4, 'distributed_rank': 0, 'batch_size': 4, 'batch_size_valid': 1}
```

</details>

Here is the full structure of the model:

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

## Eval_lm configs

The full set of configs used for the evaluation task is available here.

<details>
<summary>Toggle here</summary>
<br>

```
cfg:
	_name        None
	common       {'_name': None, 'no_progress_bar': False, 'log_interval': 100, 'log_format': None, 'log_file': None, 'tensorboard_logdir': None, 'wandb_project': None, 'azureml_logging': False, 'seed': 1, 'cpu': False, 'tpu': False, 'bf16': False, 'memory_efficient_bf16': False, 'fp16': True, 'memory_efficient_fp16': False, 'fp16_no_flatten_grads': False, 'fp16_init_scale': 128, 'fp16_scale_window': None, 'fp16_scale_tolerance': 0.0, 'min_loss_scale': 0.0001, 'threshold_loss_scale': None, 'user_dir': None, 'empty_cache_freq': 0, 'all_gather_list_size': 16384, 'model_parallel_size': 1, 'quantization_config_path': None, 'profile': False, 'reset_logging': False, 'suppress_crashes': False, 'use_plasma_view': False, 'plasma_path': '/tmp/plasma', 'log_nvidia_smi': False}
	common_eval  {'_name': None, 'path': '/mnt/en_moe_lm_15b/model.pt', 'post_process': None, 'quiet': False, 'model_overrides': "{'world_size': 4, 'moe_eval_capacity_token_fraction': 0.05}", 'results_path': None, 'is_moe': True}
	distributed_training {'_name': None, 'distributed_world_size': 4, 'distributed_rank': 0, 'distributed_backend': 'nccl', 'distributed_init_method': 'tcp://localhost:14094', 'distributed_port': -1, 'device_id': 0, 'distributed_no_spawn': False, 'ddp_backend': 'pytorch_ddp', 'bucket_cap_mb': 25, 'fix_batches_to_gpus': False, 'find_unused_parameters': False, 'fast_stat_sync': False, 'heartbeat_timeout': -1, 'broadcast_buffers': False, 'slowmo_momentum': None, 'slowmo_algorithm': 'LocalSGD', 'localsgd_frequency': 3, 'nprocs_per_node': 4, 'pipeline_model_parallel': False, 'pipeline_balance': None, 'pipeline_devices': None, 'pipeline_chunks': 0, 'pipeline_encoder_balance': None, 'pipeline_encoder_devices': None, 'pipeline_decoder_balance': None, 'pipeline_decoder_devices': None, 'pipeline_checkpoint': 'never', 'zero_sharding': 'none', 'fp16': True, 'memory_efficient_fp16': False, 'tpu': False, 'no_reshard_after_forward': False, 'fp32_reduce_scatter': False, 'cpu_offload': False, 'use_sharded_state': False, 'distributed_num_procs': 4}
	dataset      {'_name': None, 'num_workers': 1, 'num_workers_valid': 0, 'skip_invalid_size_inputs_valid_test': False, 'max_tokens': None, 'batch_size': 1, 'required_batch_size_multiple': 8, 'required_seq_len_multiple': 1, 'dataset_impl': None, 'data_buffer_size': 10, 'train_subset': 'train', 'valid_subset': 'valid', 'combine_valid_subsets': None, 'ignore_unused_valid_subsets': False, 'validate_interval': 1, 'validate_interval_updates': 0, 'validate_after_updates': 0, 'fixed_validation_seed': None, 'disable_validation': False, 'max_tokens_valid': None, 'batch_size_valid': 1, 'max_valid_steps': 4, 'curriculum': 0, 'gen_subset': 'valid', 'num_shards': 1, 'shard_id': 0}
	optimization {'_name': None, 'max_epoch': 0, 'max_update': 0, 'stop_time_hours': 0.0, 'clip_norm': 0.0, 'sentence_avg': False, 'update_freq': [1], 'lr': [0.25], 'stop_min_lr': -1.0, 'use_bmuf': False}
	checkpoint   {'_name': None, 'save_dir': 'checkpoints', 'restore_file': 'checkpoint_last.pt', 'finetune_from_model': None, 'reset_dataloader': False, 'reset_lr_scheduler': False, 'reset_meters': False, 'reset_optimizer': False, 'optimizer_overrides': '{}', 'save_interval': 1, 'save_interval_updates': 0, 'keep_interval_updates': -1, 'keep_last_epochs': -1, 'keep_best_checkpoints': -1, 'no_save': False, 'no_epoch_checkpoints': False, 'no_last_checkpoints': False, 'no_best_checkpoints': False, 'no_save_optimizer_state': False, 'no_save_optimizer_state_on_training_finished': False, 'symlink_best_and_last_checkpoints': False, 'best_checkpoint_metric': 'loss', 'maximize_best_checkpoint_metric': False, 'patience': -1, 'checkpoint_suffix': '', 'checkpoint_shard_count': 1, 'load_checkpoint_on_all_dp_ranks': False, 'write_checkpoints_asynchronously': False, 's3_upload_path': None, 'model_parallel_size': 1}
	bmuf {'_name': None, 'block_lr': 1.0, 'block_momentum': 0.875, 'global_sync_iter': 50, 'warmup_iterations': 500, 'use_nbm': False, 'average_sync': False, 'distributed_world_size': 4}
	generation   {'_name': None, 'beam': 5, 'nbest': 1, 'max_len_a': 0.0, 'max_len_b': 200, 'min_len': 1, 'match_source_len': False, 'unnormalized': False, 'no_early_stop': False, 'no_beamable_mm': False, 'lenpen': 1.0, 'unkpen': 0.0, 'replace_unk': None, 'sacrebleu': False, 'score_reference': False, 'prefix_size': 0, 'no_repeat_ngram_size': 0, 'sampling': False, 'sampling_topk': -1, 'sampling_topp': -1.0, 'constraints': None, 'temperature': 1.0, 'diverse_beam_groups': -1, 'diverse_beam_strength': 0.5, 'diversity_rate': -1.0, 'print_alignment': None, 'print_step': False, 'lm_path': None, 'lm_weight': 0.0, 'iter_decode_eos_penalty': 0.0, 'iter_decode_max_iter': 10, 'iter_decode_force_max_iter': False, 'iter_decode_with_beam': 1, 'iter_decode_with_external_reranker': False, 'retain_iter_history': False, 'retain_dropout': False, 'retain_dropout_modules': None, 'decoding_format': None, 'no_seed_provided': False}
	eval_lm      {'_name': None, 'output_word_probs': True, 'output_word_stats': False, 'context_window': 0, 'softmax_batch': 9223372036854775807, 'stats_path': None, 'max_valid_steps': 4}
	interactive  {'_name': None, 'buffer_size': 0, 'input': '-'}
	model        None
	task {'_name': 'language_modeling', 'data': '/mnt/data-bin/wikitext-103', 'sample_break_mode': 'none', 'tokens_per_sample': 2048, 'output_dictionary_size': -1, 'self_target': False, 'future_target': False, 'past_target': False, 'add_bos_token': False, 'max_source_positions': None, 'max_target_positions': None, 'shorten_method': 'none', 'shorten_data_split_list': '', 'pad_to_fixed_length': False, 'pad_to_fixed_bsz': False, 'seed': 1, 'batch_size': 1, 'batch_size_valid': 1, 'dataset_impl': None, 'data_buffer_size': 10, 'tpu': False, 'use_plasma_view': False, 'plasma_path': '/tmp/plasma'}
	criterion    {'_name': 'cross_entropy', 'sentence_avg': True}
	optimizer    None
	lr_scheduler {'_name': 'fixed', 'force_anneal': None, 'lr_shrink': 0.1, 'warmup_updates': 0, 'lr': [0.25]}
	scoring      {'_name': 'bleu', 'pad': 1, 'eos': 2, 'unk': 3}
	bpe  None
	tokenizer    None
	simul_type   None
```

</details>
