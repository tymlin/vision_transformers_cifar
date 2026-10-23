import torch
from PIL import Image
from timm.data.auto_augment import rand_augment_transform
from timm.data.random_erasing import RandomErasing
from torchvision.transforms import v2

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_basic_transform(image_size: tuple[int, int] | None = None) -> v2.Compose:
    steps: list[v2.Transform] = [v2.ToImage()]

    if image_size is not None:
        steps.append(v2.Resize(image_size, antialias=True))

    steps.append(v2.ToDtype(torch.float32, scale=True))

    transform = v2.Compose(steps)
    return transform


def build_standard_transform(image_size: tuple[int, int] | None = None) -> list[v2.Transform]:
    image_size = image_size if image_size is not None else (32, 32)
    steps: list[v2.Transform] = [
        v2.ToImage(),
        v2.Resize(image_size, antialias=True),
        v2.RandomCrop(image_size, padding=4),
        v2.RandomHorizontalFlip(p=0.5),
    ]
    return steps


def build_randaugment_transform(image_size: tuple[int, int] | None = None) -> v2.Compose:
    transform = v2.Compose(
        [
            *build_standard_transform(image_size),
            v2.RandAugment(num_ops=2, magnitude=9),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )
    return transform


def build_autoaugment_transform(image_size: tuple[int, int] | None = None) -> v2.Compose:
    transform = v2.Compose(
        [
            *build_standard_transform(image_size),
            v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )
    return transform


def build_threeaugment_transform(image_size: tuple[int, int] | None = None) -> v2.Compose:
    transform = v2.Compose(
        [
            *build_standard_transform(image_size),
            v2.RandomChoice(
                [
                    v2.Grayscale(num_output_channels=3),
                    v2.RandomSolarize(threshold=128, p=1.0),
                    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                ]
            ),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )
    return transform


def build_augmix_transform(image_size: tuple[int, int] | None = None) -> v2.Compose:
    transform = v2.Compose(
        [
            *build_standard_transform(image_size),
            v2.AugMix(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )
    return transform


def build_timm_randaugment_transform(
    image_size: tuple[int, int] | None = None,
) -> v2.Compose:
    image_size = image_size if image_size is not None else (32, 32)

    randaugment = rand_augment_transform(
        config_str="rand-m9-mstd0.5-inc1",
        hparams={
            "translate_const": int(min(image_size) * 0.45),
            "img_mean": tuple(round(255 * value) for value in CIFAR10_MEAN),
            "interpolation": Image.Resampling.BICUBIC,
            "magnitude_max": 9.99,
        },
    )

    transform = v2.Compose(
        [
            *build_standard_transform(image_size),
            v2.ToPILImage(),
            randaugment,
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            RandomErasing(probability=0.25, mode="pixel", device="cpu"),
        ]
    )
    return transform


def build_batch_transform(
    mixup_enabled: bool,
    cutmix_enabled: bool,
    num_classes: int,
) -> v2.RandomChoice | None:
    steps: list[v2.Transform] = []

    if mixup_enabled:
        steps.append(v2.MixUp(num_classes=num_classes, alpha=0.8))

    if cutmix_enabled:
        steps.append(v2.CutMix(num_classes=num_classes, alpha=1.0))

    transform = v2.RandomChoice(steps) if steps else None
    return transform


def build_eval_transform(image_size: tuple[int, int] | None = None) -> v2.Compose:
    transform = v2.Compose(
        [
            build_basic_transform(image_size),
            v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )
    return transform
