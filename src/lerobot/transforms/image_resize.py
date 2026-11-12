"""共享图像缩放：保留原始数据，在训练加载与策略预处理时等比补边。"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def letterbox_image(image: Tensor, size: tuple[int, int]) -> Tensor:
    """将 (..., C, H, W) 缩放并居中补黑边；uint8 输入统一转换为 [0, 1] 浮点数。

    尺寸使用高、宽顺序，奇数补边的多余像素位于右侧或下侧。已经匹配目标尺寸时不插值。
    """
    if len(size) != 2 or any(type(value) is not int or value <= 0 for value in size):
        raise ValueError(f"Expected positive integer height/width, got {size}")
    if image.ndim < 3:
        raise ValueError(f"Expected (..., C, H, W) image, got {tuple(image.shape)}")
    if image.dtype == torch.uint8:
        image = image.to(torch.float32) / 255.0
    height, width = image.shape[-2:]
    if (height, width) == tuple(size):
        return image
    scale = min(size[0] / height, size[1] / width)
    resized = (max(1, min(size[0], round(height * scale))), max(1, min(size[1], round(width * scale))))
    # 插值接受四维输入，因此暂时合并批次、时间等前导维度。
    leading_shape = image.shape[:-3]
    image = F.interpolate(
        image.reshape(-1, *image.shape[-3:]),
        size=resized,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    pad_h, pad_w = size[0] - resized[0], size[1] - resized[1]
    image = F.pad(image, (pad_w // 2, (pad_w + 1) // 2, pad_h // 2, (pad_h + 1) // 2))
    return image.reshape(*leading_shape, image.shape[-3], *size)


@dataclass
class LetterboxImageTransform:
    """在 DataLoader 内先缩小图像，再执行可选增强，避免整批 1080p 图像占用 GPU。"""

    size: tuple[int, int]
    augmentation: Callable[[Tensor], Tensor] | None = None

    def __call__(self, images: Tensor) -> Tensor:
        images = letterbox_image(images, self.size)
        return self.augmentation(images) if self.augmentation is not None else images
