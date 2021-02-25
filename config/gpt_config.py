# 12 layer, 12 head, 768 embed
kogpt2_config_112m_half = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 64,
    "n_embd": 768,
    "n_head": 6,
    "n_layer": 6,
    "n_positions": 64,
    "vocab_size": 32000,
    "activation_function": "gelu"
}


# 12 layer, 12 head, 768 embed
kogpt2_config_112m = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 128,
    "n_embd": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 128,
    "vocab_size": 32000,
    "resid_pdrop": 0,
    "embd_pdrop": 0,
    "attn_pdrop": 0,
    "activation_function": "gelu"
}

# 24 layer, 16 head, 1024 embed,
kogpt2_config_345m = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 64,
    "n_embd": 1024,
    "n_head": 16,
    "n_layer": 24,
    "n_positions": 64,
    "vocab_size": 32000,
    "activation_function": "gelu"
}