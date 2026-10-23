import torch
from torch import Tensor


def img_to_patch(
    x: Tensor,
    patch_size: int,
    flatten_channels: bool = True,
) -> Tensor:
    """Converts an image tensor to patches.

    :param x: torch.Tensor - input image tensor
    :param patch_size: int - size of the patch
    :param flatten_channels: bool - whether to flatten the channels or not;
                             If True, the patches will be returned in a flattened format
                             as a feature vector instead of an image grid.
    :return: torch.Tensor - tensor of patches, returns:
                            batchs size, num of patches (seq lenght), num of features (embedding size),
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    if flatten_channels:
        x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x

    # B, C, H, W = x.shape  # batch, channels, height, width
    # x_unf = torch.nn.functional.unfold(
    #     x,
    #     kernel_size=patch_size,
    #     stride=patch_size,
    # )  # [B, C*patch_size*patch_size, H*W//patch_size//patch_size]
    # x_unf = x_unf.view(B, C, -1, x_unf.size(-1))  # [B, C,  patch_size*patch_size, H*W//patch_size//patch_size]
    # x_unf = x_unf.permute(0, 3, 1, 2)  # [B, patch_size*patch_size, C, H*W//patch_size//patch_size]
    # if flatten_channels:
    #     x_unf = x_unf.reshape(B, x_unf.size(1), -1)  # [B, patch_size*patch_size, C*H*W//patch_size//patch_size]
    # return x_unf
