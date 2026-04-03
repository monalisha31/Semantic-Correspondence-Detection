import yaml
import os


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_config(config_path: str = None) -> dict:
    if config_path is None:
        candidates = [
            "configs/default.yaml",
            "../configs/default.yaml",
            os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
        ]
        for c in candidates:
            if os.path.exists(c):
                config_path = c
                break

    if config_path and os.path.exists(config_path):
        return load_config(config_path)

    print("Warning: Config file not found, using hardcoded defaults.")
    return _default_config()


def _default_config() -> dict:
    return {
        "seed": 42,
        "text_simclr": {
            "encoder_name": "all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "hidden_dim": 512,
            "projection_dim": 256,
            "batch_size": 64,
            "epochs": 15,
            "learning_rate": 3e-4,
            "weight_decay": 1e-5,
            "temperature": 0.1,
            "warmup_epochs": 2,
            "model_out": "checkpoints/text_simclr_head.pth",
        },
        "graph_simclr": {
            "node_feature_dim": 384,
            "hidden_dim": 128,
            "projection_dim": 128,
            "num_gnn_layers": 3,
            "batch_size": 32,
            "epochs": 15,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "temperature": 0.1,
            "dropout": 0.3,
            "edge_drop_rate": 0.2,
            "feature_mask_rate": 0.2,
            "model_out": "checkpoints/graph_simclr.pth",
        },
        "fusion": {
            "text_dim": 256,
            "graph_dim": 128,
            "fused_dim": 256,
            "gate_hidden": 128,
            "batch_size": 64,
            "epochs": 10,
            "learning_rate": 5e-4,
            "temperature": 0.1,
            "model_out": "checkpoints/fusion_model.pth",
        },
        "augmentations": {
            "case_variation_prob": 0.2,
            "rename_elements_prob": 0.3,
            "shuffle_lists_prob": 0.3,
            "delete_token_prob": 0.1,
            "crop_prob": 0.1,
            "synonym_replace_prob": 0.2,
            "type_generalize_prob": 0.2,
            "drop_optional_prob": 0.15,
        },
        "evaluation": {
            "k_values": [1, 3, 5, 10],
            "test_split": 0.2,
            "val_split": 0.1,
            "num_seeds": 3,
        },
    }
