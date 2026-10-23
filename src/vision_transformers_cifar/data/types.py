from enum import StrEnum, auto


class DatasetSplit(StrEnum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()
