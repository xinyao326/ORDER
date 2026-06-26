config_dict = {
    'order': {
        "backbone": "cnn",
        "n_epochs": 200,
        "lr": 3e-4,
        "dropout": 0,
        "weight_decay": 0,
        "image_size": 224,
        "hidden_dim": 512,
        "common_dim": 128,
        "latent_dim": 128,
    },

    'mech': {
        "latent_dim": 128,
        "hidden_dim": 128,
        "num_layers": 3
    },

    'prior': {
        "dim": 128,
        "depth": 6,
        "dim_head": 64,
        "heads": 8,
        "image_embed_dim": 128,
        "timesteps": 1000,
        "cond_drop_prob": 0.2
    },

    'decoder': {
        "dim": 128,
        "image_embed_dim": 128,
        "cond_dim": 128,
        "channels": 3,
        "dim_mults": (1, 2, 4, 8),
        "image_size": 224,
        "timesteps": 1000,
        "image_cond_drop_prob": 0.1,
        "text_cond_drop_prob": 0.5
    },
}