import numpy as np
import os

def load_configs(path):
    def check_int(s):
        if s[0] in ('-', '+'):
            return s[1:].isdigit()
        return s.isdigit()
    def is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False
    def is_bool(s):
        if type(s) != str:
            return False
        return s.lower() == "true" or s.lower() == "false"
    def convert_bool(s):
        assert(is_bool(s))
        return s.lower() == "true"
    configs={}
    with open(path, "r") as c:
        for line in c.readlines():
            if "=" not in line:
                continue
            l = line.strip().replace(" ", "").split("=")
            if "[" in l[1]:
                assert("]" in l[1])
                configs[l[0]] = eval(l[1])
            else:
                if check_int(l[1]):
                    configs[l[0]] = int(l[1])
                elif is_float(l[1]):
                    configs[l[0]] = float(l[1])
                elif is_bool(l[1]):
                    configs[l[0]] = convert_bool(l[1])
                else:
                    configs[l[0]] = l[1]
    return configs

folder = "/home/ubuntu/nlp_gpt3_text-generation_0.35B_MoE-64/model/c-models/1-gpu"
config_ini_filepath = os.path.join(folder, "config.ini")
conf = load_configs(config_ini_filepath)

float_type = np.half if conf["weight_data_type"] == "fp16" else np.single
global_hidden_units = conf["head_num"] * conf["size_per_head"]
max_pos_seq_len = conf["max_pos_seq_len"]
vocab_size = conf["vocab_size"]
num_layers = conf["num_layer"]
tensor_para_size = conf["tensor_para_size"]
moe_layers = conf["moe_layers"]
local_head_num = conf["head_num"] // tensor_para_size
local_hidden_units = local_head_num * conf["size_per_head"]
local_inter_size = local_hidden_units * 4
expert_num = conf["expert_num"]

wpe_shape = [max_pos_seq_len, global_hidden_units]
wte_shape = [vocab_size, global_hidden_units]
layernorm_shape = [global_hidden_units,]
attn_qkv_weight_shape = [global_hidden_units, local_hidden_units * 3]
attn_qkv_bias_shape = [local_hidden_units * 3,]
attn_out_weight_shape = [local_hidden_units, global_hidden_units]
attn_out_bias_shape = [global_hidden_units,]
ffn_h_to_4h_weight_shape = [global_hidden_units, local_inter_size]
ffn_bias1_shape = [local_inter_size]
ffn_4h_to_h_weight_shape = [local_inter_size, global_hidden_units]
ffn_bias2_shape = [global_hidden_units,]
print(wpe_shape)
print(wte_shape)
print(layernorm_shape)
print(attn_qkv_weight_shape)
print(attn_qkv_bias_shape)
print(attn_out_weight_shape)
print(attn_out_bias_shape)
print(ffn_h_to_4h_weight_shape)
print(ffn_bias1_shape)
print(ffn_4h_to_h_weight_shape)
print(ffn_bias2_shape)

wte = np.fromfile(os.path.join(folder, "model.wte.bin"), dtype=float_type)
assert(len(wte) == np.prod(wte_shape))
wte.reshape(wte_shape)
wpe = np.fromfile(os.path.join(folder, "model.wpe.bin"), dtype=float_type)
assert(len(wpe) == np.prod(wpe_shape))
wpe.reshape(wpe_shape)
final_layernorm_bias = np.fromfile(os.path.join(folder, "model.final_layernorm.bias.bin"), dtype=float_type)
assert(len(final_layernorm_bias) == np.prod(layernorm_shape))
final_layernorm_bias.reshape(layernorm_shape)
final_layernorm_weight = np.fromfile(os.path.join(folder, "model.final_layernorm.weight.bin"), dtype=float_type)
assert(len(final_layernorm_weight) == np.prod(layernorm_shape))
final_layernorm_weight.reshape(layernorm_shape)

for n_l in range(num_layers):
    attn_out_proj_bias = np.fromfile(os.path.join(folder, f"model.layers.{n_l}.attention.dense.bias.bin"), dtype=float_type)
    assert(len(attn_out_proj_bias) == np.prod(attn_out_bias_shape))
    attn_out_proj_bias.reshape(attn_out_bias_shape)
    for n_p in range(tensor_para_size):
        attn_out_proj_weight = np.fromfile(os.path.join(folder, f"model.layers.{n_l}.attention.dense.weight.{n_p}.bin"), dtype=float_type)
        assert(len(attn_out_proj_weight) == np.prod(attn_out_weight_shape))
        attn_out_proj_weight.reshape(attn_out_weight_shape)
        attn_qkv_proj_bias = np.fromfile(os.path.join(folder, f"model.layers.{n_l}.attention.query_key_value.bias.{n_p}.bin"), dtype=float_type)
        assert(len(attn_qkv_proj_bias) == np.prod(attn_qkv_bias_shape))
        attn_qkv_proj_bias.reshape(attn_qkv_bias_shape)
        attn_qkv_proj_weight = np.fromfile(os.path.join(folder, f"model.layers.{n_l}.attention.query_key_value.weight.{n_p}.bin"), dtype=float_type)
        assert(len(attn_qkv_proj_weight) == np.prod(attn_qkv_weight_shape))
        attn_qkv_proj_weight.reshape(attn_qkv_weight_shape)
        
        if n_l not in moe_layers:
            mlp_dense_h_to_4h_weight = np.fromfile(os.path.join(folder, f"model.layers.{n_l}.mlp.dense_h_to_4h.weight.{n_p}.bin"), dtype=float_type)
            assert(len(mlp_dense_h_to_4h_weight) == np.prod(ffn_h_to_4h_weight_shape))
            mlp_dense_h_to_4h_weight.reshape(ffn_h_to_4h_weight_shape)
            mlp_dense_h_to_4h_bias = np.fromfile(os.path.join(folder, f"model.layers.{n_l}.mlp.dense_h_to_4h.bias.{n_p}.bin"), dtype=float_type)
            assert(len(mlp_dense_h_to_4h_bias) == np.prod(ffn_bias1_shape))
            mlp_dense_h_to_4h_bias.reshape(ffn_bias1_shape)

            mlp_dense_4h_to_h_weight = np.fromfile(os.path.join(folder, f"model.layers.{n_l}.mlp.dense_4h_to_h.weight.{n_p}.bin"), dtype=float_type)
            assert(len(mlp_dense_4h_to_h_weight) == np.prod(ffn_4h_to_h_weight_shape))
            mlp_dense_4h_to_h_weight.reshape(ffn_4h_to_h_weight_shape)
            
        else:
            mlp_moe_experts_h_to_4h_weight = np.fromfile(os.path.join(folder, f"model.layers.{n_l}.mlp.moe.experts.dense_h_to_4h.weight.{n_p}.bin"), dtype=float_type)
            assert(len(mlp_moe_experts_h_to_4h_weight) == expert_num*np.prod(ffn_h_to_4h_weight_shape))
            mlp_moe_experts_h_to_4h_weight.reshape([expert_num] + ffn_h_to_4h_weight_shape)
            mlp_moe_experts_h_to_4h_bias = np.fromfile(os.path.join(folder, f"model.layers.{n_l}.mlp.moe.experts.dense_h_to_4h.bias.{n_p}.bin"), dtype=float_type)
            assert(len(mlp_moe_experts_h_to_4h_bias) == expert_num * np.prod(ffn_bias1_shape))
            mlp_moe_experts_h_to_4h_bias.reshape([expert_num] + ffn_bias1_shape)

            mlp_moe_experts_4h_to_h_weight = np.fromfile(os.path.join(folder, f"model.layers.{n_l}.mlp.moe.experts.dense_4h_to_h.weight.{n_p}.bin"), dtype=float_type)
            assert(len(mlp_moe_experts_4h_to_h_weight) == expert_num * np.prod(ffn_4h_to_h_weight_shape))
            mlp_moe_experts_4h_to_h_weight.reshape([expert_num] + ffn_4h_to_h_weight_shape)
            
    if n_l not in moe_layers:
        mlp_dense_4h_to_h_bias = np.fromfile(os.path.join(folder, f"model.layers.{n_l}.mlp.dense_4h_to_h.bias.bin"), dtype=float_type)
        assert(len(mlp_dense_4h_to_h_bias) == np.prod(ffn_bias2_shape))
        mlp_dense_4h_to_h_bias.reshape(ffn_bias2_shape)
        
    else:
        mlp_dense_4h_to_h_bias = np.fromfile(os.path.join(folder, f"model.layers.{n_l}.mlp.moe.experts.dense_4h_to_h.bias.bin"), dtype=float_type)
        assert(len(mlp_dense_4h_to_h_bias) == expert_num * np.prod(ffn_bias2_shape))
        mlp_dense_4h_to_h_bias.reshape([expert_num] + ffn_bias2_shape)

    input_layernorm_bias = np.fromfile(os.path.join(folder, f"model.layers.{n_l}.input_layernorm.bias.bin"), dtype=float_type)
    assert(len(input_layernorm_bias) == np.prod(layernorm_shape))
    input_layernorm_bias.reshape(layernorm_shape)
    input_layernorm_weight = np.fromfile(os.path.join(folder, f"model.layers.{n_l}.input_layernorm.weight.bin"), dtype=float_type)
    assert(len(input_layernorm_weight) == np.prod(layernorm_shape))
    input_layernorm_weight.reshape(layernorm_shape)
    post_attn_layernorm_bias = np.fromfile(os.path.join(folder, f"model.layers.{n_l}.post_attention_layernorm.bias.bin"), dtype=float_type)
    assert(len(post_attn_layernorm_bias) == np.prod(layernorm_shape))
    post_attn_layernorm_bias.reshape(layernorm_shape)
    post_attn_layernorm_weight = np.fromfile(os.path.join(folder, f"model.layers.{n_l}.post_attention_layernorm.weight.bin"), dtype=float_type)
    assert(len(post_attn_layernorm_weight) == np.prod(layernorm_shape))
    post_attn_layernorm_weight.reshape(layernorm_shape)
