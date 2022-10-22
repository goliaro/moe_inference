# Docs
This folder contains our documentation.

## Benchmarking

### Inference frameworks
We have documentation on the other inference frameworks we are benchmarking. More specifically:

- Fairseq platform: [fairseq_moe](./fairseq_moe.md).
- DeepSpeed-MoE platform: [deepspeed_moe](./deepspeed_moe.md)
- FasterTransformer platform: [faster_transformer_moe](./faster_transformer_moe.md). *Note that FasterTransformer does not natively support MoE*

### Generating Inference Request traces

The docs on how we generate the traces of inference requests, as well as the solutions from how other existing works: [inference_request_traces.md](./inference_request_traces.md)

### MoE architectures
We have documentation on the MoE models we are using for inference. Fairseq implements MoE using the G-Shard model, for which we have a summary doc. We also have docs on Fairseq's 15B-parameters LM MoE model.

* G-Shard architecture: [g-shard](./g-shard.md)
* Fairseq's 15B-parameters LM MoE checkpoint: [en_moe_lm_15b.md](./en_moe_lm_15b.md)

## Implementation
We are currently planning on building our framework on top of Legion and we may reuse parts of FlexFlow.

### Legion
For an overview of Legion: [legion.md](./legion.md)
