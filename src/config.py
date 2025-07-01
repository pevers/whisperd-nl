"""
Configuration file for Whisper Large-v3 fine-tuning
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Base training configuration parameters for Whisper Large-v3"""

    model_name: str = "openai/whisper-large-v3"
    language: str = "dutch"
    task: str = "transcribe"
    freeze_feature_encoder: bool = False
    freeze_encoder: bool = False

    # Data settings
    dataset_path: str = "data/training"
    dataset_file: str = "whisper_dataset.json"
    max_duration_seconds: float = 30.0
    min_duration_seconds: float = 0.5
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    preprocessed_cache_dir: str | None = "data/training/.cache/preprocessed"
    dataset_seed: int = 1337

    # Training hyperparameters
    output_dir: str = "./data/whisperd-nl-prod"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # Effective batch size of 16
    learning_rate: float = 5e-6  # Lower learning rate for large model
    warmup_steps: int = 500
    max_steps: int = 4500

    # Training settings
    gradient_checkpointing: bool = True
    bf16: bool = True
    dataloader_num_workers: int = 4

    # Evaluation and logging
    eval_strategy: str = "steps"
    eval_steps: int = 750
    save_steps: int = 750
    logging_steps: int = 25
    save_total_limit: int = 3

    # Generation settings
    generation_max_length: int = 448

    # Monitoring
    report_to: list | None = None
    push_to_hub: bool = False
    run_name: str | None = None

    def __post_init__(self):
        self.report_to = ["tensorboard"]


@dataclass
class QuickTestConfig(TrainingConfig):
    """Quick test configuration for whisper-large-v3 with low memory and small dataset"""

    output_dir: str = "data/whisperd-nl-test"
    max_steps: int = 1000
    eval_steps: int = 100
    save_steps: int = 100
    max_train_samples: int = 1000
    max_eval_samples: int = 100
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-6
    warmup_steps: int = 50
    dataloader_num_workers: int = 8
    run_name: str = "whisperd-nl-quick-test"
    preprocessed_cache_dir: str | None = "data/training/.cache/preprocessed-quick-test"


@dataclass
class ProductionConfig(TrainingConfig):
    """Production configuration for whisper-large-v3 with optimized parameters"""

    output_dir: str = "./data/whisperd-nl-prod"
    max_steps: int = 20000
    eval_steps: int = 4000
    save_steps: int = 4000
    learning_rate: float = 5e-6
    warmup_steps: int = 500
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 6
    gradient_accumulation_steps: int = 16
    dataloader_num_workers: int = 16
    run_name: str = "whisperd-nl-prod"


# Default configuration
DEFAULT_CONFIG = QuickTestConfig()
