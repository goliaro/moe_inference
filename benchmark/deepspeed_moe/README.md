# DeepSpeed-MoE

DeepSpeed Inference Tutorial: [link here](https://www.deepspeed.ai/tutorials/inference-tutorial/)

DeepSpeed-MoE Inference Tutorial: [link here](https://www.deepspeed.ai/tutorials/mixture-of-experts-inference/)


DeepSpeedExample: [link here](https://github.com/microsoft/DeepSpeedExamples)

Megatron-DeepSpeed: [link here](https://github.com/microsoft/Megatron-DeepSpeed#downloading-checkpoints)


Run single GPT2 inference: 
```
cd /DeepSpeedExamples/inference/huggingface/text-generation/
deepspeed --num_gpus 4 test-gpt2.py
``` 