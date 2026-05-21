import os

def get_env_bool(name, default=False):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in ["true", "1", "yes", "y"]

def get_env_int(name, default):
    value = os.getenv(name)
    return int(value) if value not in [None, ""] else default

def get_env_float(name, default):
    value = os.getenv(name)
    return float(value) if value not in [None, ""] else default

def get_env_str(name, default):
    value = os.getenv(name)
    return value if value not in [None, ""] else default

def get_env_list(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return [item.strip() for item in value.split(",")]

def apply_env_overrides(config):
    # run_train.sh에서 source "$ENV_FILE"로 이미 환경변수를 export했기 때문에
    # python-dotenv 없이 os.getenv()만 사용한다.

    # Model
    config.model.name = get_env_str("MODEL_NAME", config.model.name)
    config.model.torch_dtype = get_env_str("TORCH_DTYPE", config.model.torch_dtype)

    if hasattr(config.model, "use_flash_attention"):
        config.model.use_flash_attention = get_env_bool(
            "USE_FLASH_ATTENTION",
            config.model.use_flash_attention,
        )

    config.model.attn_implementation = get_env_str(
        "ATTN_IMPLEMENTATION",
        config.model.attn_implementation,
    )

    if hasattr(config.model, "trust_remote_code"):
        config.model.trust_remote_code = get_env_bool(
            "TRUST_REMOTE_CODE",
            config.model.trust_remote_code,
        )
    
    # LoRA
    config.model.use_lora = get_env_bool("USE_LORA", config.model.use_lora)
    config.model.lora_rank = get_env_int("LORA_RANK", config.model.lora_rank)
    config.model.lora_alpha = get_env_int("LORA_ALPHA", config.model.lora_alpha)
    config.model.lora_dropout = get_env_float("LORA_DROPOUT", config.model.lora_dropout)

    if hasattr(config.model, "lora_target_modules"):
        config.model.lora_target_modules = get_env_list(
            "LORA_TARGET_MODULES",
            config.model.lora_target_modules,
        )

    #Training
    config.training.learning_rate = get_env_float(
        "LEARNING_RATE",
        config.training.learning_rate,
    )
    config.training.weight_decay = get_env_float(
        "WEIGHT_DECAY",
        config.training.weight_decay,
    )
    config.training.warmup_ratio = get_env_float(
        "WARMUP_RATIO",
        config.training.warmup_ratio,
    )
    config.training.gradient_accumulation_steps = get_env_int(
        "GRADIENT_ACCUMULATION_STEPS",
        config.training.gradient_accumulation_steps,
    )

    if hasattr(config.training, "gradient_checkpointing"):
        config.training.gradient_checkpointing = get_env_bool(
            "GRADIENT_CHECKPOINTING",
            config.training.gradient_checkpointing,
        )

    # Data
    if hasattr(config, "data") and hasattr(config.data, "target_longest_image_dim"):
        config.data.target_longest_image_dim = get_env_int(
            "TARGET_LONGEST_IMAGE_DIM",
            config.data.target_longest_image_dim,
        )

    return config
