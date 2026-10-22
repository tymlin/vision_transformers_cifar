from enum import StrEnum, auto


class Device(StrEnum):
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"
