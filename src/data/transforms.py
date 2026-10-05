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
) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.3),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-10, 10),
                p=0.4,
            ),
            A.Resize(height=img_size[0],width= img_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
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
