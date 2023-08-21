from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import numpy
import torch
from torch import Tensor

# =========================================== ARGS VALIDATION ===========================================


def _validate_rate_curve(curve: Tensor, nan_allowed: bool = False):
    if not isinstance(curve, Tensor):
        raise ValueError(f"Expected argument `curve` to be a Tensor, but got {type(curve)}.")

    if curve.ndim != 1:
        raise ValueError(f"Expected argument `curve` to be a 1D tensor, but got {curve.ndim}D tensor.")

    if not torch.is_floating_point(curve):
        raise ValueError(f"Expected argument `curve` to have dtype float, but got {curve.dtype}.")

    if not nan_allowed:
        if torch.isnan(curve).any():
            raise ValueError("Expected argument `curve` to not contain NaN values, but got NaN values.")
        valid_values = curve
    else:
        valid_values = curve[~torch.isnan(curve)]

    if (valid_values < 0).any() or (valid_values > 1).any():
        raise ValueError(
            "Expected argument `curve` to have values in the interval [0, 1], but got values outside this interval."
        )

    diffs = curve.diff()
    diffs_valid = diffs if not nan_allowed else diffs[~torch.isnan(diffs)]

    if (diffs_valid > 0).any():
        raise ValueError(
            "Expected argument `curve` to be monotonically decreasing, but got non-monotonically decreasing values."
        )


def _validate_perimg_rate_curves(curves: Tensor, nan_allowed: bool = False):
    if not isinstance(curves, Tensor):
        raise ValueError(f"Expected argument `curves` to be a Tensor, but got {type(curves)}.")

    if curves.ndim != 2:
        raise ValueError(f"Expected argument `curves` to be a 2D tensor, but got {curves.ndim}D tensor.")

    if not torch.is_floating_point(curves):
        raise ValueError(f"Expected argument `curves` to have dtype float, but got {curves.dtype}.")

    if not nan_allowed:
        if torch.isnan(curves).any():
            raise ValueError("Expected argument `curves` to not contain NaN values, but got NaN values.")
        valid_values = curves
    else:
        valid_values = curves[~torch.isnan(curves)]

    if (valid_values < 0).any() or (valid_values > 1).any():
        raise ValueError(
            "Expected argument `curves` to have values in the interval [0, 1], but got values outside this interval."
        )

    diffs = curves.diff(dim=1)
    diffs_valid = diffs if not nan_allowed else diffs[~torch.isnan(diffs)]

    if (diffs_valid > 0).any():
        raise ValueError(
            "Expected argument `curves` to be monotonically decreasing, but got non-monotonically decreasing values."
        )


def _validate_thresholds(thresholds: Tensor):
    if not isinstance(thresholds, Tensor):
        raise ValueError(f"Expected argument `thresholds` to be a Tensor, but got {type(thresholds)}.")

    if thresholds.ndim != 1:
        raise ValueError(f"Expected argument `thresholds` to be a 1D tensor, but got {thresholds.ndim}D tensor.")

    if not torch.is_floating_point(thresholds):
        raise ValueError(f"Expected argument `thresholds` to have dtype float, but got {thresholds.dtype}.")

    diffs = thresholds.diff()
    if (diffs <= 0).any():
        raise ValueError("Expected argument `thresholds` to be strictly increasing (thresholds[k+1] > thresholds[k]), ")


def _validate_image_classes(image_classes: Tensor):
    if not isinstance(image_classes, Tensor):
        raise ValueError(f"Expected argument `image_classes` to be a Tensor, but got {type(image_classes)}.")

    if image_classes.ndim != 1:
        raise ValueError(f"Expected argument `image_classes` to be a 1D tensor, but got {image_classes.ndim}D tensor.")

    if torch.is_floating_point(image_classes):
        raise ValueError(
            "Expected argument `image_classes` to be an int or long tensor with ground truth labels, "
            f"but got a float tensor with values {image_classes.dtype}."
        )

    unique_values = torch.unique(image_classes)
    if torch.any((unique_values != 0) & (unique_values != 1)):
        raise ValueError(
            "Expected argument `image_classes` to be a *binary* tensor with ground truth labels, "
            f"but got a tensor with values {unique_values}."
        )


def _validate_and_convert_aucs(aucs: Tensor | Sequence, nan_allowed: bool = False):
    """TODO rename to 'rates' with nonzero nonone allowed"""
    if isinstance(aucs, Sequence):
        aucs = torch.as_tensor(aucs)

    if not isinstance(aucs, Tensor):
        raise ValueError(f"Expected argument `aucs` to be a Tensor or Sequence, but got {type(aucs)}.")

    if aucs.ndim != 1:
        raise ValueError(f"Expected argument `aucs` to be a 1D tensor, but got {aucs.ndim}D tensor.")

    if not torch.is_floating_point(aucs):
        raise ValueError(f"Expected argument `aucs` to have dtype float, but got {aucs.dtype}.")

    valid_aucs = aucs[~torch.isnan(aucs)] if nan_allowed else aucs

    if torch.any((valid_aucs < 0) | (valid_aucs > 1)):
        raise ValueError("Expected argument `aucs` to be in [0, 1], but got values outside this range.")

    return aucs


def _validate_image_class(image_class: int | None):
    if image_class is None:
        return

    if not isinstance(image_class, int):
        raise ValueError(f"Expected argument `image_class` to be either None or an int, but got {type(image_class)}.")

    if image_class not in (0, 1):
        raise ValueError(
            "Expected argument `image_class` to be either 0, 1 or None (respec., 'normal', 'anomalous', or 'both') "
            f"but got {image_class}."
        )


def _validate_atleast_one_anomalous_image(image_classes: Tensor):
    if (image_classes == 1).sum() == 0:
        raise ValueError("Expected argument at least one anomalous image, but found none.")


def _validate_atleast_one_normal_image(image_classes: Tensor):
    if (image_classes == 0).sum() == 0:
        raise ValueError("Expected argument at least one normal image, but found none.")


def _validate_and_convert_rate(rate: float | int | Tensor, nonzero: bool = True, nonone: bool = False) -> Tensor:
    """Validate a rate (e.g. FPR, TPR) to be in [0, 1], (0, 1), [0, 1), or (0, 1] and convert it to a tensor.
    nonzero/nonone defaults are True/False for backward compatibility.
    """

    if isinstance(rate, (float, int)):
        rate = torch.as_tensor(rate)

    elif not isinstance(rate, Tensor):
        raise ValueError(f"Expected argument to be a float, int, or torch.Tensor, but got {type(rate)}.")

    if rate.dim() != 0:
        raise ValueError(f"Expected argument to be a scalar, but got a tensor of shape {rate.shape}.")

    if rate < 0 or rate > 1:
        raise ValueError(f"Argument `{rate}` is not a valid because it is <0 or >1.")

    if nonzero and rate == 0:
        raise ValueError("Expected argument to be > 0.")

    if nonone and rate == 1:
        raise ValueError("Expected argument to be < 1.")

    return rate


def _validate_and_convert_threshold(threshold: float | int | Tensor) -> Tensor:
    """Validate a threshold and convert it to a tensor."""

    if isinstance(threshold, (float, int)):
        threshold = torch.as_tensor(threshold)

    elif not isinstance(threshold, Tensor):
        raise ValueError(f"Expected argument to be a float, int, or torch.Tensor, but got {type(threshold)}.")

    if threshold.dim() != 0:
        raise ValueError(f"Expected argument to be a scalar, but got a tensor of shape {threshold.shape}.")

    return threshold


def _validate_and_convert_fpath(fpath: str | Path, extension: str | None) -> Path:
    if isinstance(fpath, str):
        fpath = Path(fpath)

    if not isinstance(fpath, Path):
        raise ValueError(f"Expected argument to be a str or Path, but got {type(fpath)}.")

    if fpath.is_dir():
        raise ValueError("Expected argument to be a file, but got a directory.")

    if extension is not None:
        if len(fpath.suffix) == 0:
            fpath = fpath.with_suffix(extension)

        elif fpath.suffix != extension:
            raise ValueError(f"Expected argument to have extension {extension}, but got {fpath.suffix}.")

    return fpath


# =========================================== FUNCTIONAL ===========================================


def perimg_boxplot_stats(
    values: Tensor, image_classes: Tensor, only_class: int | None = None
) -> list[dict[str, str | int | float | None]]:
    """Compute boxplot statistics for a given tensor of values.

    This function uses `matplotlib.cbook.boxplot_stats`, which is the same function used by `matplotlib.pyplot.boxplot`.

    Args:
        values (Tensor): Tensor of per-image values.
        image_classes (Tensor): Tensor of image classes.
        only_class (int | None): If not None, only compute statistics for images of the given class.
                                 None means both image classes are used. Defaults to None.

    Returns:
        list[dict[str, str | int | float | None]]: List of boxplot statistics.
        Each dictionary has the following keys:
            - 'statistic': Name of the statistic.
            - 'value': Value of the statistic (same units as `values`).
            - 'nearest': Some statistics (e.g. 'mean') are not guaranteed to be in the tensor, so this is the
                            closest to the statistic in an actual image (i.e. in `values`).
            - 'imgidx': Index of the image in `values` that has the `nearest` value to the statistic.
    """

    _validate_image_classes(image_classes)
    _validate_image_class(only_class)

    if values.ndim != 1:
        raise ValueError(f"Expected argument `values` to be a 1D tensor, but got {values.ndim}D tensor.")

    if values.shape != image_classes.shape:
        raise ValueError(
            "Expected arguments `values` and `image_classes` to have the same shape, "
            f"but got {values.shape} and {image_classes.shape}."
        )

    if only_class is not None and only_class not in image_classes:
        raise ValueError(f"Argument `only_class` is {only_class}, but `image_classes` does not contain this class.")

    # convert to numpy because of `matplotlib.cbook.boxplot_stats`
    values = values.cpu().numpy()
    image_classes = image_classes.cpu().numpy()

    # only consider images of the given class
    imgs_mask = numpy.ones_like(image_classes, dtype=bool) if only_class is None else (image_classes == only_class)
    values = values[imgs_mask]
    imgs_idxs = numpy.nonzero(imgs_mask)[0]

    def arg_find_nearest(stat_value):
        return (numpy.abs(values - stat_value)).argmin()

    # function used in `matplotlib.boxplot`
    boxplot_stats = mpl.cbook.boxplot_stats(values)[0]  # [0] is for the only boxplot

    records = []

    def append_record(stat_, val_):
        # make sure to use a value that is actually in the array
        # because some statistics (e.g. 'mean') are not guaranteed to be in the array
        invalues_idx = arg_find_nearest(val_)
        nearest = values[invalues_idx]
        imgidx = imgs_idxs[invalues_idx]
        records.append(
            dict(
                statistic=stat_,
                value=val_,
                nearest=nearest,
                imgidx=imgidx,
            )
        )

    for stat, val in boxplot_stats.items():
        if stat in ("iqr", "cilo", "cihi"):
            continue

        elif stat != "fliers":
            append_record(stat, val)
            continue

        for val_ in val:
            append_record(
                "flierhi" if val_ > boxplot_stats["med"] else "flierlo",
                val_,
            )

    records = sorted(records, key=lambda r: r["value"])
    return records


def _validate_and_convert_models_dict(models: dict[str, Tensor | Sequence]):
    """
    `models` is expected to be a dict of tensors of shape (num_images,) with
    the per-image metric value \\in [0, 1] for each image.

    key: model name
    value: tensor of shape (num_images,)

    if they have `nan` values, all of them must have `nan` at the same indices.
    i.e. if one tensor has `nan` at index i, all tensors must have `nan` at index i

    if the values are sequences, they are converted to tensors
    """

    if not isinstance(models, dict):
        raise TypeError(f"Expected argument `models` to be a dict, but got {type(models)}.")

    if len(models) < 2:
        raise ValueError("Expected argument `models` to have at least one key, but got none.")

    # make sure all keys are strings (the model names)
    if not all(isinstance(k, str) for k in models.keys()):
        raise TypeError("Expected argument `models` to have all keys of type str.")

    tmp = {}
    for k in models.keys():
        try:
            tmp[k] = _validate_and_convert_aucs(models[k], nan_allowed=True)
        except Exception as ex:
            raise ValueError(
                f"Expected argument `models` to have all sequences of floats \\in [0, 1]. Key {k}."
            ) from ex
    models = tmp

    unique_shapes = sorted({t.shape for t in models.values()})
    if len(unique_shapes) != 1:
        raise ValueError(f"Expected argument `models` to have all tensors of the same shape. But found {unique_shapes}")

    # make sure all tensors have `nan` at the same indices
    # nan values mask of an arbitrary tensor
    nan_mask = torch.isnan(next(iter(models.values())))
    for t in models.values():
        if (torch.isnan(t) != nan_mask).any():
            raise ValueError("Expected argument `models` to have all tensors with `nan` at the same indices.")

    return models
