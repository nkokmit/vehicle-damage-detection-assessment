"""Transform helpers for vehicle damage preprocessing and augmentation."""

from typing import Any, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(
    img_size: tuple[int, int] = (800, 800),
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    use_advanced_aug: bool = True,
) -> A.Compose:
    """Build Albumentations training augmentation pipeline.

    Args:
        img_size: Target (height, width).
        mean: Normalization channel means.
        std: Normalization channel standard deviations.
        use_advanced_aug: If True, apply specular-reflection-robust augmentations
                          (ColorJitter, RandomGamma, GaussNoise, GaussianBlur, CoarseDropout).
    """
    ops = [A.HorizontalFlip(p=0.5)]

    if use_advanced_aug:
        # Augmentation chuyên biệt chống ảo giác ánh sáng chói & bóng đèn phản chiếu trên sơn bóng
        ops.extend([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussNoise(std_range=(0.1, 0.3), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(num_holes_range=(2, 6), hole_height_range=(0.04, 0.10), hole_width_range=(0.04, 0.10), p=0.3),
        ])
    else:
        ops.extend([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.3),
        ])

    ops.extend([
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.05, 0.05),
            rotate=(-10, 10),
            p=0.4,
        ),
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return A.Compose(
        ops,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["category_ids"],
            min_visibility=0.0,
        ),
    )


def get_val_transforms(
    img_size: tuple[int, int] = (800, 800),
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> A.Compose:
    return A.Compose(
        [
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["category_ids"],
            min_visibility=0.0,
        ),
    )


def get_default_transform() -> Callable[[Any], Any]:
    return lambda item: item
