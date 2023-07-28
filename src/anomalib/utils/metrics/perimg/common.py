from __future__ import annotations

import matplotlib as mpl
import numpy
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.pyplot import Figure
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


def _validate_aucs(aucs: Tensor, nan_allowed: bool = False):
    if not isinstance(aucs, Tensor):
        raise ValueError(f"Expected argument `aucs` to be a Tensor, but got {type(aucs)}.")

    if aucs.ndim != 1:
        raise ValueError(f"Expected argument `aucs` to be a 1D tensor, but got {aucs.ndim}D tensor.")

    if not torch.is_floating_point(aucs):
        raise ValueError(f"Expected argument `aucs` to have dtype float, but got {aucs.dtype}.")

    valid_aucs = aucs[~torch.isnan(aucs)] if nan_allowed else aucs

    if torch.any((valid_aucs < 0) | (valid_aucs > 1)):
        raise ValueError("Expected argument `aucs` to be in [0, 1], but got values outside this range.")


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


# =========================================== FUNCTIONAL ===========================================


def _perimg_boxplot_stats(
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
                         It is None if the statistic is in the tensor.
            - 'imgidx': Index of the image in `values` that has the `nearest` value to the statistic.

            `nearest` and `imgidx`:
                - If `value == values[imgidx]` (i.e. this will happen with quartiles), this is None.
                - If `value` != values[imgidx]` (i.e. this will, most likely, happen with the mean),
                    this is the value in `values` that is closest to `value`.

                In both cases, `imgidx` is always valid.
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

    if only_class not in image_classes:
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
                nearest=nearest if nearest != val_ else None,  # None if the value is in the array
                imgidx=imgidx,
            )
        )

    for stat, val in boxplot_stats.items():
        if stat == "iqr":
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


# =========================================== PLOT UTILS ===========================================


def _bounded_lims(ax: Axes, axis: int, bounds: tuple[float | None, float | None] = (None, None)):
    """Snap X/Y-axis limits to stay within the given bounds."""

    assert len(bounds) == 2, f"Expected argument `bounds` to be a tuple of size 2, but got size {len(bounds)}."

    if axis == 0:
        lims = ax.get_xlim()
    elif axis == 1:
        lims = ax.get_ylim()
    else:
        raise ValueError(f"Unknown axis {axis}. Must be 0 (X-axis) or 1 (Y-axis).")

    newlims = list(lims)

    if bounds[0] is not None and lims[0] < bounds[0]:
        newlims[0] = bounds[0]

    if bounds[1] is not None and lims[1] > bounds[1]:
        newlims[1] = bounds[1]

    if axis == 0:
        ax.set_xlim(newlims)
    else:
        ax.set_ylim(newlims)


# =========================================== PLOT ===========================================


def _plot_perimg_metric_boxplot(
    values: Tensor,
    image_classes: Tensor,
    annotate: bool = True,
    only_class: int | None = None,
    ax: Axes | None = None,
) -> tuple[Figure | None, Axes]:
    _validate_image_classes(image_classes)
    _validate_image_class(only_class)

    if values.ndim != 1:
        raise ValueError(f"Expected argument `values` to be a 1D tensor, but got {values.ndim}D tensor.")

    if values.shape != image_classes.shape:
        raise ValueError(
            "Expected arguments `values` and `image_classes` to have the same shape, "
            f"but got {values.shape} and {image_classes.shape}."
        )

    if only_class not in image_classes:
        raise ValueError(f"Argument `only_class` is {only_class}, but `image_classes` does not contain this class.")

    fig, ax = plt.subplots() if ax is None else (None, ax)

    # only consider images of the given class
    imgs_mask = (
        torch.ones_like(image_classes, dtype=torch.bool) if only_class is None else (image_classes == only_class)
    )

    ax.boxplot(
        values[imgs_mask],
        vert=False,
        widths=0.5,
        showmeans=True,
        showcaps=True,
    )
    _ = ax.set_yticks([])

    if annotate:
        bp_stats = _perimg_boxplot_stats(values, image_classes, only_class=only_class)
        num_images = len(values)
        num_flierlo = len([s for s in bp_stats if s["statistic"] == "flierlo"])
        num_flierhi = len([s for s in bp_stats if s["statistic"] == "flierhi"])
        ax.annotate(
            text=f"Number of images\n    total: {num_images}\n    outliers: {num_flierlo} low, {num_flierhi} high",
            xy=(0.03, 0.95),
            xycoords="axes fraction",
            xytext=(0, 0),
            textcoords="offset points",
            annotation_clip=False,
            verticalalignment="top",
        )

    return fig, ax
