import torch
from torchvision.transforms import v2


def build_transform(image_size: tuple[int, int] | None = None) -> v2.Compose:
    steps: list[v2.Transform] = [v2.ToImage()]

    if image_size is not None:
        steps.append(v2.Resize(image_size, antialias=True))

    steps.append(v2.ToDtype(torch.float32, scale=True))

    return v2.Compose(steps)
