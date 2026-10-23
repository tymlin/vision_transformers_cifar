from abc import ABC, abstractmethod
from typing import Any

import torch
import torchinfo
from torch import nn


class BaseModel(nn.Module, ABC):
    name: str

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        raise NotImplementedError()

    def summary(self, example_input_shape: tuple[int, int, int, int], depth: int = 4) -> str:
        col_names = ["input_size", "output_size", "num_params", "params_percent"]
        model_summary = str(
            torchinfo.summary(
                self,
                input_size=example_input_shape,
                depth=depth,
                col_names=col_names,
                verbose=0,
                device=self.device,
            )
        )
        return model_summary
