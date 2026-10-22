from enum import StrEnum, auto


class DatasetSplit(StrEnum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()


class DatasetType(StrEnum):
    CIFAR10 = auto()
