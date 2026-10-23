from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vision_transformers_cifar.consts import RUNS_DIRPATH
from vision_transformers_cifar.data.types import DatasetSplit
from vision_transformers_cifar.loggers import logger
from vision_transformers_cifar.modules.types import Accelerator, DeviceSelection, OptimizerType, Precision


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True, allow_inf_nan=False)


class DataConfig(BaseConfig):
    train_split: DatasetSplit = Field(
        default=DatasetSplit.TRAIN,
        description="Source dataset split used for training. Must be 'train'.",
    )
    val_split: DatasetSplit = Field(
        default=DatasetSplit.TEST,
        description=(
            "Source dataset split used for validation. Use 'test' for datasets without a dedicated validation split."
        ),
    )
    test_split: DatasetSplit | None = Field(
        default=None,
        description="Source dataset split used for testing. None means no separate test dataset is created.",
    )
    image_size: tuple[Annotated[int, Field(ge=1)], Annotated[int, Field(ge=1)]] | None = Field(
        default=None,
        description="Target image size as (height, width). None preserves the original image size.",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Number of samples per batch for training, validation and testing.",
    )
    num_workers: int = Field(
        default=8,
        ge=0,
        description="Number of subprocesses used for data loading. Zero loads data in the main process.",
    )
    persistent_workers: bool = Field(
        default=True,
        description=(
            "Whether to keep data loading workers alive between epochs. Requires `num_workers` greater than zero."
        ),
    )
    pin_memory: bool = Field(
        default=True,
        description=(
            "Whether to copy loaded tensors into pinned host memory to support faster transfers to CUDA devices."
        ),
    )
    drop_last: bool = Field(
        default=True,
        description=(
            "Whether to drop the final training batch when it contains fewer samples than batch_size. "
            "Does not affect validation or testing."
        ),
    )
    shuffle: bool = Field(
        default=True,
        description="Whether to shuffle training samples each epoch. Does not affect validation or testing.",
    )

    @model_validator(mode="after")
    def validate_workers(self) -> Self:
        if self.persistent_workers and self.num_workers == 0:
            msg = f"Persistent workers require num_workers > 0; got {self.num_workers}."
            logger.error(msg)
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_splits(self) -> Self:
        if self.train_split != DatasetSplit.TRAIN:
            msg = f"Training requires the train split; got {self.train_split!r}."
            logger.error(msg)
            raise ValueError(msg)
        if self.val_split not in (DatasetSplit.VALIDATION, DatasetSplit.TEST):
            msg = f"Validation requires the validation or test split; got {self.val_split!r}."
            logger.error(msg)
            raise ValueError(msg)
        if self.test_split not in (None, DatasetSplit.TEST):
            msg = f"Testing requires the test split or None; got {self.test_split!r}."
            logger.error(msg)
            raise ValueError(msg)
        return self


class ViTModelConfig(BaseConfig):
    embed_dim: int = Field(
        default=256,
        ge=1,
        description="Token embedding dimension. Must be divisible by num_heads.",
    )
    hidden_dim: int = Field(
        default=512,
        ge=1,
        description="Hidden dimension of the feed-forward network in each transformer block.",
    )
    num_heads: int = Field(
        default=8,
        ge=1,
        description="Number of attention heads in each transformer block.",
    )
    num_layers: int = Field(
        default=6,
        ge=1,
        description="Number of transformer blocks.",
    )
    patch_size: int = Field(
        default=4,
        ge=1,
        description="Height and width of each image patch in pixels.",
    )
    num_channels: int = Field(
        default=3,
        ge=1,
        description="Number of input image channels.",
    )
    num_patches: int = Field(
        default=64,
        ge=1,
        description="Maximum number of image patches supported by the positional embeddings, excluding the CLS token.",
    )
    num_classes: int = Field(
        default=10,
        ge=1,
        description="Number of output classes.",
    )
    dropout: float = Field(
        default=0.2,
        ge=0,
        lt=1,
        description="Dropout probability for the input embeddings and feed-forward networks.",
    )

    @model_validator(mode="after")
    def validate_attention_heads(self) -> Self:
        if self.embed_dim % self.num_heads:
            msg = f"embed_dim={self.embed_dim} must be divisible by num_heads={self.num_heads}."
            logger.error(msg)
            raise ValueError(msg)
        return self


class OptimizerConfig(BaseConfig):
    name: OptimizerType = Field(
        default=OptimizerType.ADAMW,
        description="Optimizer used to update model parameters: AdamW or Adam.",
    )
    lr: float = Field(
        default=3e-4,
        gt=0,
        description="Learning rate used by the optimizer.",
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0,
        description="Weight decay coefficient used by the optimizer.",
    )
    betas: tuple[Annotated[float, Field(ge=0, lt=1)], Annotated[float, Field(ge=0, lt=1)]] = Field(
        default=(0.9, 0.999),
        description="Exponential decay rates for the first and second moment estimates.",
    )
    eps: float = Field(
        default=1e-8,
        gt=0,
        description="Small constant added to the optimizer denominator for numerical stability.",
    )


class SchedulerConfig(BaseConfig):
    enabled: bool = Field(
        default=True,
        description="Enable learning rate scheduling. False keeps the optimizer learning rate constant.",
    )


class TrainerConfig(BaseConfig):
    max_epochs: int = Field(
        default=10,
        ge=1,
        description="Number of training epochs.",
    )
    accelerator: Accelerator = Field(
        default=Accelerator.AUTO,
        description="Execution accelerator. Auto lets Lightning select an available accelerator.",
    )
    devices: Annotated[int, Field(ge=-1)] | DeviceSelection | list[Annotated[int, Field(ge=0)]] = Field(
        default=1,
        description="Device count, list of device indices, -1 for all devices, or 'auto' for automatic selection.",
    )
    precision: Precision = Field(
        default=Precision.FLOAT32,
        description="Numerical precision used by Lightning. Defaults to full 32-bit precision.",
    )
    log_every_n_steps: int = Field(
        default=1,
        ge=1,
        description="Number of training steps between metric logging updates.",
    )
    accumulate_grad_batches: int = Field(
        default=1,
        ge=1,
        description="Number of training batches to accumulate before each optimizer step.",
    )
    limit_train_batches: Annotated[int, Field(ge=0)] | Annotated[float, Field(ge=0, le=1)] = Field(
        default=1.0,
        description="Training batches per epoch: an integer count or a float fraction between zero and one.",
    )
    limit_val_batches: Annotated[int, Field(ge=0)] | Annotated[float, Field(ge=0, le=1)] = Field(
        default=1.0,
        description="Validation batches per run: an integer count or a float fraction between zero and one.",
    )
    limit_test_batches: Annotated[int, Field(ge=0)] | Annotated[float, Field(ge=0, le=1)] = Field(
        default=1.0,
        description="Test batches per run: an integer count or a float fraction between zero and one.",
    )

    @model_validator(mode="after")
    def validate_devices(self) -> Self:
        if self.devices == 0:
            msg = f"`devices` must be positive, -1, 'auto', or a nonempty list of indices; got {self.devices}."
            logger.error(msg)
            raise ValueError(msg)
        if isinstance(self.devices, list) and not self.devices:
            msg = f"`devices` must contain at least one device index; got {self.devices}."
            logger.error(msg)
            raise ValueError(msg)
        return self


class LoggingConfig(BaseConfig):
    csv: bool = Field(
        default=True,
        description="Save metrics to CSV. Enables metric plots generated from that CSV after training.",
    )
    tensorboard: bool = Field(
        default=True,
        description="Save metrics to TensorBoard event files.",
    )


class RunConfig(BaseConfig):
    run_dirpath: Path = Field(
        default=RUNS_DIRPATH,
        description="Directory used to store run outputs.",
    )
    run_name_suffix: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional name appended to the timestamp in the run directory name. None means only the timestamp is used."
        ),
    )
    seed: int = Field(
        default=42,
        ge=0,
        le=2**32 - 1,
        description="Random seed used by Python, NumPy, PyTorch and training data shuffling.",
    )
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Dataset selection, source splits and data loading settings.",
    )
    model: ViTModelConfig = Field(
        default_factory=ViTModelConfig,
        description="Arguments used to construct the Vision Transformer model.",
    )
    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig,
        description="Optimizer hyperparameters.",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="Learning rate scheduler settings.",
    )
    trainer: TrainerConfig = Field(
        default_factory=TrainerConfig,
        description="Training duration, device, visualization and metric settings.",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Enable or disable CSV and TensorBoard metric logging.",
    )

    @model_validator(mode="after")
    def validate_model_input(self) -> Self:
        height, width = self.data.image_size or (32, 32)
        patch_size = self.model.patch_size
        if height % patch_size or width % patch_size:
            msg = f"Image size {(height, width)} must be divisible by patch_size={patch_size}."
            logger.error(msg)
            raise ValueError(msg)
        num_patches = (height // patch_size) * (width // patch_size)
        if self.model.num_patches < num_patches:
            msg = f"The model needs at least {num_patches} positional patch embeddings for this image size."
            logger.error(msg)
            raise ValueError(msg)
        if self.model.num_channels != 3 or self.model.num_classes != 10:
            msg = (
                f"CIFAR-10 requires num_channels=3 and num_classes=10; "
                f"got {self.model.num_channels=} and {self.model.num_classes=}."
            )
            logger.error(msg)
            raise ValueError(msg)
        return self
