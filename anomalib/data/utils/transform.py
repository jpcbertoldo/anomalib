"""Helper function for retrieving transforms."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import logging
from enum import Enum
from typing import Optional, Tuple, Union

import albumentations as A
from albumentations.pytorch import ToTensorV2

from anomalib.data.utils.image import get_image_height_and_width

logger = logging.getLogger(__name__)


class InputNormalizationMethod(str, Enum):
    """Normalization method for the input images."""

    NONE = "none"  # no normalization applied
    IMAGENET = "imagenet"  # normalization to ImageNet statistics

    @staticmethod
    def get_transform(input_normalization_method: "InputNormalizationMethod") -> A.Compose:
        """Get normalization transform.

        Args:
            input_normalization_method (InputNormalizationMethod)

        Returns:
            A.Compose: Normalization transform with pre-defined parameters.

        Raises:
            ValueError: When normalization method is not recognized.
        """
        if input_normalization_method == InputNormalizationMethod.IMAGENET:
            return A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
        elif input_normalization_method == InputNormalizationMethod.NONE:
            return A.ToFloat(max_value=255)  # FIXME this is a misleading name, it should be ToFloat255 or something
        
        else:
            raise ValueError(f"Unknown normalization method: {input_normalization_method}")


class PredefinedTranforms(str, Enum):
    """Predefined transforms for the input images."""
    
    IMAGENET = "imagenet"
    GAUSSIANAD = "gaussian-ad"
    

def build_tranform_gaussian_ad(
    flip, rotate90, background_edge, rotate45, 
    augment_probability, image_size, normalization, to_tensor,
):
    """Get transforms for GaussianAD.
    
    augment_probability=0.5 is used in the original code by default.    
      
    Base on 
    https://github.com/ORippler/gaussian-ad-mvtec/blob/bc10bd736d85b750410e6b0e7ac843061e09511e/src/common/augmentation.py
    
    Changes due to deprecated functions:
    (old) --> (new)
    # IAASharpen --> Sharpen
    # IAAEmboss --> Emboss
    # IAAAdditiveGaussianNoise --> GaussNoise
    
    """

    transforms_list = []

    # resize image
    if image_size is not None:
        resize_height, resize_width = get_image_height_and_width(image_size)
        transforms_list.append(A.Resize(height=resize_height, width=resize_width, always_apply=True))

    augmentations = []

    # pixel-wise augmentations
    # based on `pixel_aug(noise=True)` in
    augmentations.extend([
        A.OneOf(
            [
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ],
            p=0.2,
        ),
        A.OneOf(
            [
                A.CLAHE(clip_limit=2),
                # A.IAASharpen(),
                A.Sharpen(),
                # A.IAAEmboss(),
                A.Emboss(),
                A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2)),
            ],
            p=0.3,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, val_shift_limit=(-10, 20), p=0.3
        ),
        A.OneOf(
            [
                # A.IAAAdditiveGaussianNoise(scale=(0.01 * 255, 0.03 * 255), per_channel=False),
                # A.IAAAdditiveGaussianNoise(scale=(0.01 * 255, 0.03 * 255), per_channel=True),
                A.GaussNoise(var_limit=(0.01 * 255, 0.03 * 255), per_channel=False),
                A.GaussNoise(var_limit=(0.01 * 255, 0.03 * 255), per_channel=True),
            ],
            p=0.2,
        ),
    ])
    
    # image-wise augmentations
    # based on `detection_aug()` in
    if flip:
        augmentations.append(A.HorizontalFlip())
        
    if rotate90:
        augmentations.append(A.RandomRotate90())
        
    augmentations.append(
        A.ShiftScaleRotate(
            shift_limit=0.05 if background_edge else 0,
            scale_limit=(-0.05, 0.1 if background_edge else 0),
            rotate_limit=(45 if rotate45 else 15) if background_edge else 0,
            p=0.2,
        )
    )
    
    if augment_probability > 0:
        # the original code used p=0.5 at this compose by default
        transforms_list.append(A.Compose(augmentations, p=augment_probability))
    
    # normalization
    transforms_list.append(InputNormalizationMethod.get_transform(normalization))

    # tensor conversion
    if to_tensor:
        transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)


def get_tranform_gaussian_ad(category, config):
    """Get transforms for GaussianAD with predefined parameters or build it from kwargs."""
    
    if category is None:
        return build_tranform_gaussian_ad(**config)
    
    # from MVTecAD.augmentation_info() from gaussian-ad-mvtec/src/datasets/mvteacad.py
    PREDEF_KWARGS_BY_CATEGORY = dict(
        bottle=dict(flip=True, rotate90=True, rotate45=False, background_edge=True),
        carpet=dict(flip=True, rotate90=True, rotate45=True, background_edge=True),
        leather=dict(flip=True, rotate90=True, rotate45=True, background_edge=True),
        pill=dict(flip=False, rotate90=True, rotate45=False, background_edge=True),
        tile=dict(flip=True, rotate90=True, rotate45=True, background_edge=True),
        wood=dict(flip=True, rotate90=True, rotate45=False, background_edge=True),
        cable=dict(flip=True, rotate90=True, rotate45=False, background_edge=True),
        grid=dict(flip=True, rotate90=True, rotate45=True, background_edge=False),
        toothbrush=dict(flip=True, rotate90=True, rotate45=False, background_edge=True),
        zipper=dict(flip=True, rotate90=True, rotate45=False, background_edge=True),
        capsule=dict(flip=False, rotate90=True, rotate45=False, background_edge=True),
        hazelnut=dict(flip=True, rotate90=True, rotate45=True, background_edge=True),
        metal_nut=dict(flip=False, rotate90=True, rotate45=False, background_edge=True),
        screw=dict(flip=False, rotate90=True, rotate45=False, background_edge=True),
        transistor=dict(flip=True, rotate90=False, rotate45=False, background_edge=True),
    )
    
    return build_tranform_gaussian_ad(**{**config, **PREDEF_KWARGS_BY_CATEGORY[category]})
        

def get_predifined_transform(predifined, config):
    """Get predefined transform."""
    
    if predifined == PredefinedTranforms.IMAGENET:
        # FIXME this is temporary, in reality there would be the code in the "else" of get_transforms
        return get_transforms(
            image_size=config['image_size'],
            center_crop=config['center_crop'],
            normalization=InputNormalizationMethod.IMAGENET,
            to_tensor=config['to_tensor'],
        )
    
    elif predifined == PredefinedTranforms.GAUSSIANAD:
        config = deepcopy(config)
        return get_tranform_gaussian_ad(category=config.pop('category', None), config=config)
    
    else:
        raise ValueError(f"Predefined transform ``{predifined}`` not recognized.")


def get_transforms(
    config: Optional[Union[str, A.Compose]] = None,
    image_size: Optional[Union[int, Tuple[int, int]]] = None,
    center_crop: Optional[Union[int, Tuple[int, int]]] = None,
    normalization: InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
    to_tensor: bool = True,
) -> A.Compose:
    """Get transforms from config or image size.

    Args:
        config (Optional[Union[str, A.Compose]], optional): Albumentations transforms.
            Either config or albumentations ``Compose`` object. Defaults to None.
        image_size (Optional[Union[int, Tuple]], optional): Image size to transform. Defaults to None.
        to_tensor (bool, optional): Boolean to convert the final transforms into Torch tensor. Defaults to True.

    Raises:
        ValueError: When both ``config`` and ``image_size`` is ``None``.
        ValueError: When ``config`` is not a ``str`` or `A.Compose`` object.

    Returns:
        A.Compose: Albumentation ``Compose`` object containing the image transforms.

    Examples:
        >>> import skimage
        >>> image = skimage.data.astronaut()

        >>> transforms = get_transforms(image_size=256, to_tensor=False)
        >>> output = transforms(image=image)
        >>> output["image"].shape
        (256, 256, 3)

        >>> transforms = get_transforms(image_size=256, to_tensor=True)
        >>> output = transforms(image=image)
        >>> output["image"].shape
        torch.Size([3, 256, 256])


        Transforms could be read from albumentations Compose object.
        >>> import albumentations as A
        >>> from albumentations.pytorch import ToTensorV2
        >>> config = A.Compose([A.Resize(512, 512), ToTensorV2()])
        >>> transforms = get_transforms(config=config, to_tensor=False)
        >>> output = transforms(image=image)
        >>> output["image"].shape
        (512, 512, 3)
        >>> type(output["image"])
        numpy.ndarray

        Transforms could be deserialized from a yaml file.
        >>> transforms = A.Compose([A.Resize(1024, 1024), ToTensorV2()])
        >>> A.save(transforms, "/tmp/transforms.yaml", data_format="yaml")
        >>> transforms = get_transforms(config="/tmp/transforms.yaml")
        >>> output = transforms(image=image)
        >>> output["image"].shape
        torch.Size([3, 1024, 1024])
    """
    transforms: A.Compose

    if config is not None:

        logger.info("``image_size``, ``center_crop``, ``normalization`` and ``to_tensor`` will be ignored because ``config`` is provided.")
        
        # load transforms from config file
        if isinstance(config, str):
            logger.info("Reading transforms from Albumentations config file: %s.", config)
            transforms = A.load(filepath=config, data_format="yaml")
            
        elif isinstance(config, A.Compose):
            logger.info("Transforms loaded from Albumentations Compose object")
            transforms = config
            
        elif isinstance(config, dict):
            config = deepcopy(config)
            predifined = config.pop("predefined")
            transforms = get_predifined_transform(predifined, config)
            
        else:
            raise ValueError("config could be either ``str`` or ``A.Compose``")
    else:
        logger.info("No config file has been provided. Using default transforms.")
        transforms_list = []

        # add resize transform
        if image_size is None:
            raise ValueError(
                "Both config and image_size cannot be `None`. "
                "Provide either config file to de-serialize transforms "
                "or image_size to get the default transformations"
            )
        resize_height, resize_width = get_image_height_and_width(image_size)
        transforms_list.append(A.Resize(height=resize_height, width=resize_width, always_apply=True))

        # add center crop transform
        if center_crop is not None:
            crop_height, crop_width = get_image_height_and_width(center_crop)
            if crop_height > resize_height or crop_width > resize_width:
                raise ValueError(f"Crop size may not be larger than image size. Found {image_size} and {center_crop}")
            transforms_list.append(A.CenterCrop(height=crop_height, width=crop_width, always_apply=True))

        # add normalize transform
        transforms_list.append(InputNormalizationMethod.get_transform(normalization))

        # add tensor conversion
        if to_tensor:
            transforms_list.append(ToTensorV2())

        transforms = A.Compose(transforms_list)

    return transforms
