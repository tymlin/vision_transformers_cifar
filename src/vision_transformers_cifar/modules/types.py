from enum import StrEnum, auto


class Accelerator(StrEnum):
    AUTO = auto()
    CPU = auto()
    CUDA = auto()
    GPU = auto()
    MPS = auto()


class DeviceSelection(StrEnum):
    AUTO = auto()


class Precision(StrEnum):
    FLOAT32 = "32-true"
    FLOAT16 = "16-true"
    BFLOAT16 = "bf16-true"
    FLOAT16_MIXED = "16-mixed"
    BFLOAT16_MIXED = "bf16-mixed"


class OptimizerType(StrEnum):
    ADAMW = auto()
    ADAM = auto()
