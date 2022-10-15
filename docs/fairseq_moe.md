# Fairseq MoE 

This document contains an analysis of what we discovered when it comes to the design, implementation, and performance of the Fairseq inference framework

## Parallelism in Fairseq MoE (inference)

Fairseq by default uses data parallelism only; model parallelism is supported by passing a `model_parallelism_size` parameter greater than 1. Model parallel is implemented using Megatron, in particular, the framework loads Megatron as a submodule using [this fork](https://github.com/ngoyal2707/Megatron-LM) (branch `fairseq`).

## Synchronization/communication collectives

TODO: figure out what synchronization collectives are used. Eg: SPMD, gang-scheduling, MPMD, etc...?

## Profiling & Benchmarking Evaluation 

So far, we have evaluated Fairseq using the [fairseq_cli/eval_lm.py](https://github.com/gabrieleoliaro/fairseq/blob/moe/fairseq_cli/eval_lm.py) script from the `fairseq` repository (branch `moe`). The original purpose of the script was to evaluate the accuracy of a language model trained with Fairseq. To do so, the script needs to perform inference for each input sample. In the future, we will likely need to modify/extend the script to focus more on throughput/accuracy benchmarking, but right now we are just trying to identify the bottlenecks of the system. In addition to `eval_lm.py`, `fairseq` also comes with other inference scripts, such as [fairseq_cli/generate.py](https://github.com/gabrieleoliaro/fairseq/blob/moe/fairseq_cli/generate.py). In the future, we may also want to compare `eval_lm.py` with `generate.py`, and figure out which script is best to use for our purposes.

### `eval_lm.py` configs

For the language modeling task (default task for the eval_lm script), the configs are available below:

<details>
<summary>**model configs** (which determine the model type/structure) </summary>
<br>

```
{'_name': 'transformer_lm_gpt', 'activation_fn': 'gelu', 'dropout': 0.1, 'attention_dropout': 0.1, 'activation_dropout': 0.0, 'relu_dropout': 0.0, 'decoder_embed_dim': 768, 'decoder_output_dim': 768, 'decoder_input_dim': 768, 'decoder_ffn_embed_dim': 3072, 'decoder_layers': 12, 'decoder_attention_heads': 12, 'decoder_normalize_before': True, 'no_decoder_final_norm': False, 'adaptive_softmax_cutoff': None, 'adaptive_softmax_dropout': 0.0, 'adaptive_softmax_factor': 4.0, 'no_token_positional_embeddings': False, 'share_decoder_input_output_embed': True, 'character_embeddings': False, 'character_filters': '[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]', 'character_embedding_dim': 4, 'char_embedder_highway_layers': 2, 'adaptive_input': False, 'adaptive_input_factor': 4.0, 'adaptive_input_cutoff': None, 'tie_adaptive_weights': False, 'tie_adaptive_proj': False, 'decoder_learned_pos': False, 'layernorm_embedding': False, 'no_scale_embedding': False, 'checkpoint_activations': True, 'offload_activations': False, 'decoder_layerdrop': 0.0, 'decoder_layers_to_keep': None, 'quant_noise_pq': 0.0, 'quant_noise_pq_block_size': 8, 'quant_noise_scalar': 0.0, 'min_params_to_wrap': 100000000, 'alternate_decoder_ffn_embed_dim': 0, 'moe_freq': 2, 'moe_expert_count': 512, 'moe_gating_use_fp32': True, 'moe_second_expert_policy': 'all', 'moe_normalize_gate_prob_before_dropping': False, 'moe_expert_ffn_dim': None, 'moe_top1_expert': False, 'moe_eval_capacity_token_fraction': 0.05, 'moe_normalize_expert_grad': 'sqrt_world_size', 'use_moe_pad_mask': False, 'record_a2a_perf_stats': False, 'dummy_a2a': False, 'use_stable_embedding': False, 'base_layers': 0, 'base_sublayers': 1, 'base_shuffle': 0, 'add_bos_token': False, 'tokens_per_sample': 2048, 'max_target_positions': 2048, 'tpu': False, 'memory_efficient_fp16': True, 'fp16': True, 'fp16_no_flatten_grads': False, 'ddp_backend': 'fully_sharded', 'world_size': 4, 'distributed_rank': 0, 'batch_size': 4, 'batch_size_valid': 1}
```

</details>

<details>
<summary>**The** **full set of configs** used for the evaluation task</summary>
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


### Profiling technique
The profiling was done as follows. Using the [Pytorch profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html), we wrapped the `eval_lm` function in the profiler context `with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:` ([see here](https://github.com/gabrieleoliaro/fairseq/blob/443be319435410ed8a63de0ae2ec25ba5cf6adaf/fairseq_cli/eval_lm.py#L320)), and tagged several functions of interest using the `with record_function("<CODE SECTION NAME>"):` directive. In particular, we tagged the `forward()` method of the following layers (please refer above for the structure of the model):
- `TransformerDecoder`
	- `TransformerDecoderLayer without MoE` (for those layers without a MoE)
		- `MultiheadAttention`
	- `TransformerDecoderLayer MoE` (for those layers with a MoE)
		- `MultiheadAttention`
		- `MoELayer` 
The inference is performed using data parallelism with N=4 threads, each using its own V100 GPU. Each thread is spawned via `torch.multiprocessing.spawn` ([see here](https://github.com/gabrieleoliaro/fairseq/blob/443be319435410ed8a63de0ae2ec25ba5cf6adaf/fairseq/distributed/utils.py#L362)). We intially tried wrapping the entire main function (before the call to spawn) into the profiler context, in the hope that the profiler could automatically gather data from the N=4 processes and merge them, but that didn't work, so we resorted to profiling each thread individually and saving the results to separate files.

### Profiling results
An initial profiling of the `fairseq_cli.eval_lm` script using [the 15B pre-trained MoE-GPT model](./en_moe_lm_15b.md), yielded the results that are available in the [results](../benchmark/fairseq/results) folder.

#### Analysis
Among the results, the two most interesting files are those recording the functions sorted by total cpu time, and those recording the functions sorted by total cuda time. For each of these two views, we have one file for each threads. The results are similar, so we will just include one file (using thread 1) for each of the two cases.

***CPU time total (thread 1)***

```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                eval_lm        28.68%        6.920s        99.41%       23.986s       23.986s       0.000us         0.00%        1.073s        1.073s    -192.10 Kb    -192.16 Kb      11.03 Mb      -3.07 Gb             1  
                                     TransformerDecoder         0.11%      27.132ms        51.31%       12.381s        2.476s       0.000us         0.00%        1.018s     203.577ms         -20 b     -32.00 Mb       1.16 Gb    -409.12 Mb             5  
                       TransformerDecoderLayer with MoE         0.34%      81.178ms        29.16%        7.036s     234.543ms       0.000us         0.00%     944.564ms      31.485ms        -120 b        -600 b     273.04 Mb     -19.13 Gb            30  
                                               MoELayer         6.01%        1.450s        28.30%        6.828s     227.591ms       0.000us         0.00%     899.195ms      29.973ms        -120 b        -584 b      18.95 Gb    -105.64 Gb            30  
                                       cudaLaunchKernel        19.62%        4.735s        19.62%        4.735s      95.159us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b         49760  
                                               aten::eq         0.03%       6.307ms        15.14%        3.652s      48.696ms     110.000us         0.00%     110.000us       1.467us           0 b           0 b      60.00 Kb      60.00 Kb            75  
                                           aten::linear         0.65%     157.940ms        15.08%        3.638s     453.959us       0.000us         0.00%     430.346ms      53.693us           0 b           0 b      13.65 Gb           0 b          8015  
                                             aten::item         2.06%     496.279ms        11.99%        2.894s      70.591us       0.000us         0.00%      41.010ms       1.000us           0 b           0 b           0 b           0 b         40995  
                              aten::_local_scalar_dense         3.56%     858.811ms         9.94%        2.398s      58.485us      41.010ms         1.40%      41.010ms       1.000us           0 b           0 b           0 b           0 b         40995  
                                            aten::addmm         4.76%        1.149s         7.19%        1.735s     224.223us     404.881ms        13.80%     404.881ms      52.310us           0 b           0 b      11.92 Gb      11.92 Gb          7740  
                                     MultiheadAttention         0.21%      49.825ms         6.80%        1.642s      27.361ms       0.000us         0.00%      84.565ms       1.409ms        -240 b      -1.17 Kb     182.25 Mb     -12.32 Gb            60  
                    TransformerDecoderLayer without MoE         0.09%      20.704ms         6.72%        1.621s      54.018ms       0.000us         0.00%      60.959ms       2.032ms        -120 b        -600 b     275.25 Mb      -2.38 Gb            30  
                                        cudaMemcpyAsync         6.43%        1.552s         6.43%        1.552s      37.688us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b         41192  
                                           aten::matmul         0.03%       8.375ms         6.18%        1.492s       5.424ms       0.000us         0.00%      22.259ms      80.942us           0 b           0 b       1.73 Gb           0 b           275  
                                               aten::mm         0.21%      51.296ms         6.17%        1.489s       4.444ms     117.239ms         4.00%     117.239ms     349.967us           0 b           0 b       4.08 Gb       4.08 Gb           335  
                                               cudaFree         5.90%        1.424s         5.90%        1.424s     356.062ms       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             4  
                                               aten::to         0.35%      84.594ms         5.14%        1.239s     149.490us       0.000us         0.00%     166.648ms      20.105us      11.00 Mb           0 b      44.05 Gb           0 b          8289  
                                         aten::_to_copy         0.84%     201.650ms         4.78%        1.155s     141.573us       0.000us         0.00%     166.648ms      20.435us      11.00 Mb           0 b      44.05 Gb           0 b          8155  
                                           aten::select         2.83%     683.034ms         3.54%     854.815ms      16.689us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b         51220  
                                          aten::reshape         0.31%      74.413ms         3.34%     806.619ms     201.655us       0.000us         0.00%      17.317ms       4.329us           0 b           0 b       2.26 Gb           0 b          4000  
                                            aten::copy_         1.80%     433.995ms         3.07%     741.312ms      60.728us     185.424ms         6.32%     187.298ms      15.343us           0 b     -11.00 Mb           0 b           0 b         12207  
                                            aten::clone         0.37%      88.486ms         2.84%     684.072ms     174.064us       0.000us         0.00%      18.635ms       4.742us           0 b           0 b       2.56 Gb           0 b          3930  
                                          aten::type_as         0.19%      45.499ms         2.53%     610.816ms     157.589us       0.000us         0.00%      46.759ms      12.064us           0 b           0 b       9.49 Gb           0 b          3876  
                                    aten::empty_strided         1.91%     460.748ms         1.97%     474.228ms      56.896us       0.000us         0.00%       0.000us       0.000us      11.00 Mb      11.00 Mb      48.58 Gb      48.58 Gb          8335  
                                             aten::add_         1.04%     250.474ms         1.76%     424.359ms      56.014us      31.290ms         1.07%      31.290ms       4.130us           0 b           0 b           0 b           0 b          7576  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 24.129s
Self CUDA time total: 2.935s
```

***CUDA time total (thread 1)***

```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
ncclKernel_SendRecv_RING_SIMPLE_Sum_int8_t(ncclWorkE...         0.00%       0.000us         0.00%       0.000us       0.000us        1.861s        63.43%        1.861s      31.025ms           0 b           0 b           0 b           0 b            60  
                                                eval_lm        28.68%        6.920s        99.41%       23.986s       23.986s       0.000us         0.00%        1.073s        1.073s    -192.10 Kb    -192.16 Kb      11.03 Mb      -3.07 Gb             1  
                                     TransformerDecoder         0.11%      27.132ms        51.31%       12.381s        2.476s       0.000us         0.00%        1.018s     203.577ms         -20 b     -32.00 Mb       1.16 Gb    -409.12 Mb             5  
                       TransformerDecoderLayer with MoE         0.34%      81.178ms        29.16%        7.036s     234.543ms       0.000us         0.00%     944.564ms      31.485ms        -120 b        -600 b     273.04 Mb     -19.13 Gb            30  
                                               MoELayer         6.01%        1.450s        28.30%        6.828s     227.591ms       0.000us         0.00%     899.195ms      29.973ms        -120 b        -584 b      18.95 Gb    -105.64 Gb            30  
                                           aten::linear         0.65%     157.940ms        15.08%        3.638s     453.959us       0.000us         0.00%     430.346ms      53.693us           0 b           0 b      13.65 Gb           0 b          8015  
                                            aten::addmm         4.76%        1.149s         7.19%        1.735s     224.223us     404.881ms        13.80%     404.881ms      52.310us           0 b           0 b      11.92 Gb      11.92 Gb          7740  
                                            aten::copy_         1.80%     433.995ms         3.07%     741.312ms      60.728us     185.424ms         6.32%     187.298ms      15.343us           0 b     -11.00 Mb           0 b           0 b         12207  
            volta_fp16_s884gemm_fp16_64x128_ldg8_f2f_tn         0.00%       0.000us         0.00%       0.000us       0.000us     178.657ms         6.09%     178.657ms      45.821us           0 b           0 b           0 b           0 b          3899  
                                               aten::to         0.35%      84.594ms         5.14%        1.239s     149.490us       0.000us         0.00%     166.648ms      20.105us      11.00 Mb           0 b      44.05 Gb           0 b          8289  
                                         aten::_to_copy         0.84%     201.650ms         4.78%        1.155s     141.573us       0.000us         0.00%     166.648ms      20.435us      11.00 Mb           0 b      44.05 Gb           0 b          8155  
            volta_fp16_s884gemm_fp16_128x64_ldg8_f2f_tn         0.00%       0.000us         0.00%       0.000us       0.000us     150.257ms         5.12%     150.257ms      39.129us           0 b           0 b           0 b           0 b          3840  
                                               aten::mm         0.21%      51.296ms         6.17%        1.489s       4.444ms     117.239ms         4.00%     117.239ms     349.967us           0 b           0 b       4.08 Gb       4.08 Gb           335  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      91.726ms         3.13%      91.726ms      23.340us           0 b           0 b           0 b           0 b          3930  
                                     MultiheadAttention         0.21%      49.825ms         6.80%        1.642s      27.361ms       0.000us         0.00%      84.565ms       1.409ms        -240 b      -1.17 Kb     182.25 Mb     -12.32 Gb            60  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      74.098ms         2.53%      74.098ms       6.366us           0 b           0 b           0 b           0 b         11640  
                                              aten::bmm         0.06%      15.680ms         0.14%      34.982ms     194.344us      73.538ms         2.51%      73.538ms     408.544us           0 b           0 b      29.94 Gb           0 b           180  
                    TransformerDecoderLayer without MoE         0.09%      20.704ms         6.72%        1.621s      54.018ms       0.000us         0.00%      60.959ms       2.032ms        -120 b        -600 b     275.25 Mb      -2.38 Gb            30  
                                             aten::gelu         0.92%     221.000ms         1.28%     308.808ms      79.795us      53.650ms         1.83%      53.650ms      13.863us           0 b           0 b      18.82 Gb      18.82 Gb          3870  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      53.650ms         1.83%      53.650ms      13.863us           0 b           0 b           0 b           0 b          3870  
void gemmk1_kernel<float, 256, 5, false, false, fals...         0.00%       0.000us         0.00%       0.000us       0.000us      51.934ms         1.77%      51.934ms     865.567us           0 b           0 b           0 b           0 b            60  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      50.616ms         1.72%      50.616ms      12.498us           0 b           0 b           0 b           0 b          4050  
                                              aten::add         0.84%     202.625ms         1.19%     287.287ms      86.820us      49.291ms         1.68%      49.291ms      14.896us          44 b          40 b      12.43 Gb      12.43 Gb          3309  
volta_fp16_s884gemm_fp16_128x128_ldg8_f2f_stages_32x...         0.00%       0.000us         0.00%       0.000us       0.000us      48.766ms         1.66%      48.766ms       1.626ms           0 b           0 b           0 b           0 b            30  
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      47.035ms         1.60%      47.035ms      25.247us           0 b           0 b           0 b           0 b          1863  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 24.129s
Self CUDA time total: 2.935s

```

### Next steps
First, we should look into whether it's possible to profile all processes jointly in a automatic manner. Perhaps it's possible to pass the profiler object from the context to the threads? 

Another next step is to enable model parallelism (using Megatron), and see what changes. 

Finally, we will want to run benchmarking at a larger scale, and get results for multiple values of the parametes.
