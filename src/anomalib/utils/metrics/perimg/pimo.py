"""Per-Image Overlap (PIMO, pronounced pee-mo) curve.

Two variants of AUCs are implemented:
    - AUPImO: Area Under the Per-Image Overlap (PIMO) curves.
              I.e. a metric of per-image average TPR.

for shared fpr = mean( perimg fpr ) == set fpr
    find the th = fpr^-1( MAX_FPR ) with a binary search on the pixels of the norm images
    i.e. it's not necessary to compute the perimg fpr curves (tf. binclf curves) in advance
for other shared fpr alternatives, it's necessary to compute the perimg fpr curves first anyway

further: also choose the th upper bound to be the max score at normal pixels
"""


from __future__ import annotations

from collections import namedtuple
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.pyplot import Figure
from matplotlib.ticker import FixedLocator, LogFormatter, PercentFormatter
from numpy import ndarray
from torch import Tensor

from .binclf_curve import PerImageBinClfCurve, _validate_atleast_one_anomalous_image, _validate_atleast_one_normal_image
from .common import (
    _bounded_lims,
    _perimg_boxplot_stats,
    _plot_perimg_metric_boxplot,
    _validate_aucs,
    _validate_image_classes,
    _validate_perimg_rate_curves,
    _validate_rate_curve,
)

# =========================================== VALIDATIONS ===========================================


def _validate_fpr_upper_bound(fpr: float | Tensor) -> None:
    if isinstance(fpr, float):
        fpr = torch.as_tensor(fpr)

    elif not isinstance(fpr, Tensor):
        raise ValueError(f"Expected argument `fpr` to be a float or torch.Tensor, but got {type(fpr)}.")

    if fpr.dim() != 0:
        raise ValueError(f"Expected argument `fpr` to be a scalar, but got a tensor of shape {fpr.shape}.")

    if fpr <= 0 or fpr > 1:
        raise ValueError(f"Expected argument `fpr` to be in (0, 1], but got {fpr}.")


def _validate_kwargs_perimg(kwargs_perimg: tuple[dict[str, Any] | None, ...], num_images: int) -> None:
    if len(kwargs_perimg) == 0:
        pass

    elif len(kwargs_perimg) != num_images:
        raise ValueError(
            f"Expected argument `kwargs_perimg` to have the same number of dicts as number of images, "
            f"but got {len(kwargs_perimg)} dicts while {num_images} images."
        )

    elif len(othertypes := {type(kws) for kws in kwargs_perimg if kws is not None and not isinstance(kws, dict)}) > 0:
        raise ValueError(
            "Expected argument `kwargs_perimg` to be a list of dicts or Nones, "
            f"but found {sorted(othertypes, key=lambda t: t.__name__)} instead."
        )


# =========================================== PLOT ===========================================


def plot_pimo_curves(
    shared_fpr: Tensor,
    tprs: Tensor,
    image_classes: Tensor,
    *kwargs_perimg: dict[str, Any | None] | None,
    # ---
    ax: Axes | None = None,
    logfpr: bool = False,
    logfpr_epsilon: float = 1e-4,
    **kwargs_shared,
) -> tuple[Figure | None, Axes]:
    """Plot shared FPR vs Per-Image Overlap (PImO) curves.

    The `image_classes` tensor is used to filter out the normal images, while making it possible to
        keep the indices of the anomalous images.

    Args:
        shared_fpr: shape (num_thresholds,)
        tprs: shape (num_images, num_thresholds)
        image_classes: shape (num_images,)
        ax: matplotlib Axes
        logfpr: whether to use log scale for the FPR axis
        logfpr_epsilon: small positive number to avoid `log(0)`; used only if `logfpr` is True

        *kwargs_perimg: keyword arguments passed to `ax.plot()` and SPECIFIC to each curve
                            if provided it should be a list of dicts of length `num_images`.
                            If None, that curve will not be ploted.

                            if provided it should be a list of dicts of length `num_images`.
                            If None, that curve will not be ploted.

        **kwargs: keyword arguments passed to `ax.plot()` and SHARED by all curves

        If both `kwargs_perimg` and `kwargs_shared` have the same key, the value in `kwargs_perimg` will be used.

    Returns:
        fig, ax
    """

    _validate_perimg_rate_curves(tprs, nan_allowed=True)  # normal images have `nan`s
    _validate_rate_curve(shared_fpr)
    _validate_image_classes(image_classes)

    # `shared_fpr` and `tprs` have the same number of thresholds
    if tprs.shape[1] != shared_fpr.shape[0]:
        raise ValueError(
            f"Expected argument `tprs` to have the same number of thresholds as argument `shared_fpr`, "
            f"but got {tprs.shape[1]} thresholds and {shared_fpr.shape[0]} thresholds, respectively."
        )

    # `tprs` and `image_classes` have the same number of images
    if tprs.shape[0] != image_classes.shape[0]:
        raise ValueError(
            f"Expected argument `tprs` to have the same number of images as argument `image_classes`, "
            f"but got {tprs.shape[0]} images and {image_classes.shape[0]} images, respectively."
        )

    # specific to TPR curves
    _validate_atleast_one_anomalous_image(image_classes)
    # there may be `nan`s but only in the normal images
    # in the curves of anomalous images, there should NOT be `nan`s
    _validate_perimg_rate_curves(tprs[image_classes == 1], nan_allowed=False)

    _validate_kwargs_perimg(kwargs_perimg, num_images=tprs.shape[0])

    fig, ax = plt.subplots(figsize=(7, 6)) if ax is None else (None, ax)

    # override defaults with user-provided values
    kwargs_shared = {
        **dict(linewidth=1, linestyle="-", alpha=0.3),
        **kwargs_shared,
    }

    for imgidx, (curve, img_cls) in enumerate(zip(tprs, image_classes)):
        if img_cls == 0:  # normal image
            continue

        # default label and shared kwargs
        kw = {**dict(label=f"idx={imgidx:03}"), **kwargs_shared}  # override sequence (left to right)

        if len(kwargs_perimg) == 0:
            pass
        elif kwargs_perimg[imgidx] is None:
            continue
        else:
            # override with image-specific kwargs
            kw_img: dict[str, Any] = kwargs_perimg[imgidx]  # type: ignore
            kw = {**kw, **kw_img}  # type: ignore
        ax.plot(shared_fpr, curve, **kw)
    ax.set_xlabel("Shared FPR")

    if logfpr:
        if logfpr_epsilon <= 0:
            raise ValueError(f"Expected argument `logfpr_epsilon` to be positive, but got {logfpr_epsilon}.")

        if logfpr_epsilon >= 1:
            raise ValueError(f"Expected argument `logfpr_epsilon` to be less than 1, but got {logfpr_epsilon}.")

        ax.set_xscale("log")
        ax.set_xlim(logfpr_epsilon, 1)
        eps_round_exponent = int(np.floor(np.log10(logfpr_epsilon)))
        ticks_major = np.logspace(eps_round_exponent, 0, abs(eps_round_exponent) + 1)
        formatter_major = LogFormatter()
        ticks_minor = np.logspace(eps_round_exponent, 0, 2 * abs(eps_round_exponent) + 1)

    else:
        XLIM_EPSILON = 0.01
        ax.set_xlim(0 - XLIM_EPSILON, 1 + XLIM_EPSILON)
        ticks_major = np.linspace(0, 1, 6)
        formatter_major = PercentFormatter(1, decimals=0)
        ticks_minor = np.linspace(0, 1, 11)

    ax.xaxis.set_major_locator(FixedLocator(ticks_major))
    ax.xaxis.set_major_formatter(formatter_major)
    ax.xaxis.set_minor_locator(FixedLocator(ticks_minor))

    ax.set_ylabel("Per-Image Overlap (in-image TPR)")
    YLIM_EPSILON = 0.01
    ax.set_ylim(0 - YLIM_EPSILON, 1 + YLIM_EPSILON)
    ax.yaxis.set_major_locator(FixedLocator(np.linspace(0, 1, 6)))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_minor_locator(FixedLocator(np.linspace(0, 1, 11)))

    ax.set_title("Per-Image Overlap Curves")

    return fig, ax


def plot_fprs_vs_shared_fpr(
    fprs: Tensor,
    shared_fpr: Tensor,
    image_classes: Tensor,
    *kwargs_perimg: dict[str, Any | None] | None,
    # ---
    ax: Axes | None = None,
    **kwargs_shared,
) -> tuple[Figure | None, Axes]:
    """Plot shared FPR vs in-image FPR curves."""

    _validate_perimg_rate_curves(fprs, nan_allowed=True)  # anomalous images may have `nan`s if all pixels are anomalous
    _validate_rate_curve(shared_fpr)
    _validate_image_classes(image_classes)

    # `shared_fpr` and `fprs` have the same number of thresholds
    if fprs.shape[1] != shared_fpr.shape[0]:
        raise ValueError(
            f"Expected argument `fprs` to have the same number of thresholds as argument `shared_fpr`, "
            f"but got {fprs.shape[1]} thresholds and {shared_fpr.shape[0]} thresholds, respectively."
        )

    # `fprs` and `image_classes` have the same number of images
    if fprs.shape[0] != image_classes.shape[0]:
        raise ValueError(
            f"Expected argument `fprs` to have the same number of images as argument `image_classes`, "
            f"but got {fprs.shape[0]} images and {image_classes.shape[0]} images, respectively."
        )

    # there may be `nan`s but only in the anomalous images
    # in the curves of normal images, there should NOT be `nan`s
    if (image_classes == 1).sum() > 0:
        _validate_perimg_rate_curves(fprs[image_classes == 1], nan_allowed=False)

    _validate_kwargs_perimg(kwargs_perimg, num_images=fprs.shape[0])

    fig, ax = plt.subplots(figsize=(7, 6)) if ax is None else (None, ax)

    # override defaults with user-provided values
    kwargs_shared = {
        **dict(linewidth=0.5, linestyle="-", alpha=0.3),
        **kwargs_shared,
    }

    for imgidx, (curve, img_cls) in enumerate(zip(fprs, image_classes)):
        default_label = f"idx={imgidx:03} " + ("(norm)" if img_cls == 0 else "(anom)")
        kw = {**dict(label=default_label), **kwargs_shared}  # override sequence (left to right)

        if len(kwargs_perimg) == 0:
            pass
        elif kwargs_perimg[imgidx] is None:
            continue
        else:
            # override with image-specific kwargs
            kw_img: dict[str, Any] = kwargs_perimg[imgidx]  # type: ignore
            kw = {**kw, **kw_img}  # type: ignore

        ax.plot(shared_fpr, curve, **kw)

    ax.set_xlabel("Shared FPR")

    XLIM_EPSILON = 0.01
    ax.set_xlim(0 - XLIM_EPSILON, 1 + XLIM_EPSILON)
    ax.xaxis.set_major_locator(FixedLocator(np.linspace(0, 1, 6)))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.xaxis.set_minor_locator(FixedLocator(np.linspace(0, 1, 11)))

    ax.set_ylabel("In-Image FPR")
    YLIM_EPSILON = 0.01
    ax.set_ylim(0 - YLIM_EPSILON, 1 + YLIM_EPSILON)
    ax.yaxis.set_major_locator(FixedLocator(np.linspace(0, 1, 6)))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_minor_locator(FixedLocator(np.linspace(0, 1, 11)))

    ax.set_title("FPR: Shared vs In-Image Curves")

    return fig, ax


def plot_fprs_vs_shared_fpr_predefviz(
    fprs: Tensor,
    shared_fpr: Tensor,
    image_classes: Tensor,
    mode: str,
    # ---
    ax: Axes | None = None,
    **kwargs_shared,
) -> tuple[Figure | None, Axes]:
    """Pre-defined visualization of the shared FPR vs in-image FPR curves.

    `mode="norm-vs-anom"`: normal images are in *blue*, anomalous images are in *red*

    `mode="norm-only"`: only normal images are plotted along with their statistics across the images --
                        it corresponds to taking (for ex) the mean along the y axis at a given x value in the plot.
                        Statistics: min(), max(), and mean() wiht 3 SEM interval.

    Args:
        mode: one of {"norm-vs-anom", "norm-only"}
        others as in `plot_fprs_vs_shared_fpr()`
    """
    _validate_image_classes(image_classes)

    if mode == "norm-vs-anom":
        _validate_atleast_one_anomalous_image(image_classes)
        _validate_atleast_one_normal_image(image_classes)
        # color the lines by the image class; normal = blue, anomalous = red
        kwargs_perimg = [dict(color="blue" if imgclass == 0 else "red", label=None) for imgclass in image_classes]
        # make a legend only show one normal and one anomalous line
        # `[0][0]`: first `[0]` is for the tuple from `np.where()`, second `[0]` is for the first index
        kwargs_perimg[np.where(image_classes == 0)[0][0]]["label"] = "normal (blue)"
        kwargs_perimg[np.where(image_classes == 1)[0][0]]["label"] = "anomalous (red)"
        fig, ax = plot_fprs_vs_shared_fpr(
            fprs,
            shared_fpr,
            image_classes,
            *kwargs_perimg,
            ax=ax,
            **kwargs_shared,
        )
        ax.set_title(ax.get_title() + " (Norm. vs Anom. Images)")
        ax.legend(loc="lower right", fontsize="small", title_fontsize="small", title="image class")
        return fig, ax

    if mode == "norm-only":
        _validate_atleast_one_normal_image(image_classes)
        # don't plot anomalous images
        kwargs_perimg = [dict(label=None) if imgclass == 0 else None for imgclass in image_classes]  # type: ignore
        fig, ax = plot_fprs_vs_shared_fpr(
            fprs,
            shared_fpr,
            image_classes,
            *kwargs_perimg,
            ax=ax,
            **kwargs_shared,
        )
        fprs_norm = fprs[image_classes == 0]
        ax.plot(
            shared_fpr, mean := fprs_norm.mean(dim=0), color="black", linewidth=2, linestyle="--", alpha=1, label="mean"
        )
        ax.plot(shared_fpr, fprs_norm.min(dim=0)[0], color="green", linewidth=2, linestyle="--", alpha=1, label="min")
        ax.plot(shared_fpr, fprs_norm.max(dim=0)[0], color="orange", linewidth=2, linestyle="--", alpha=1, label="max")
        ax.set_title(ax.get_title() + " (Norm. Images Only)")
        sem = fprs.std(dim=0) / torch.sqrt(torch.tensor(fprs.shape[0]))
        ax.fill_between(
            shared_fpr, mean - 3 * sem, mean + 3 * sem, color="black", alpha=0.5, label="3 SEM (mean's 99% CI)"
        )
        ax.legend(loc="lower right", fontsize="small", title_fontsize="small", title="Stats across images")
        return fig, ax

    raise ValueError(f"Expected argument `mode` to be one of {{'norm-vs-anom', 'norm-only'}}, but got {mode}.")


def plot_pimo_curves_of_boxplot_stats(
    # same as `plot_pimo_curves()`
    shared_fpr,
    tprs,
    image_classes,
    # new
    aupimo_boxplot_stats: list[dict[str, str | int | float | None]],
    # same
    ax: Axes | None = None,
    logfpr: bool = False,
    logfpr_epsilon: float = 1e-4,
    # same
    **kwargs_shared,
) -> tuple[Figure | None, Axes]:
    """Plot shared FPR vs Per-Image Overlap (PImO) curves only for the boxplot stats cases.

    Args:
        arguments not mentioned here are as in `plot_pimo_curves()`
        Refer to `anomalib.utils.metrics.perimg.pimo.plot_pimo_curves()`

        This one does not have the argument `kwargs_perimg` because it's used to plot the curves of individual
            images only with their corresponding statistic and hide the other curves.

        aupimo_boxplot_stats: list of dicts, each dict is a boxplot stat of AUPImO values
                                refer to `anomalib.utils.metrics.perimg.common._perimg_boxplot_stats()`

    Returns:
        fig, ax
    """

    if len(aupimo_boxplot_stats) == 0:
        raise ValueError("Expected argument `aupimo_boxplot_stats` to have at least one dict, but got none.")

    # it is sorted so that only the first one has a label (others are plotted but don't show in the legend)
    imgidxs_toplot_fliers: list[int] = sorted(
        {s["imgidx"] for s in aupimo_boxplot_stats if s["statistic"] in ("flierlo", "flierhi")}  # type: ignore
    )
    imgidxs_toplot_others = {s["imgidx"] for s in aupimo_boxplot_stats if s["statistic"] not in ("flierlo", "flierhi")}

    kwargs_perimg = []

    # it's not necessary to validate (shared_fpr, tprs, image_classes) because
    # they are validated in `plot_pimo_curves()` and only used for this:
    num_images = len(image_classes)
    for imgidx in range(num_images):
        if imgidx in imgidxs_toplot_fliers:
            kw = dict(linewidth=0.5, color="gray", alpha=0.8, linestyle="--")
            # only one of them will show in the legend
            if imgidx == imgidxs_toplot_fliers[0]:
                kw["label"] = "flier"
            else:
                kw["label"] = None
            kwargs_perimg.append(kw)
            continue

        if imgidx not in imgidxs_toplot_others:
            # don't plot this curve
            kwargs_perimg.append(None)  # type: ignore
            continue

        imgidx_stats = [s for s in aupimo_boxplot_stats if s["imgidx"] == imgidx]
        stat_dict = imgidx_stats[0]

        # edge case where more than one stat falls on the same image
        if len(imgidx_stats) > 1:
            stat_dict["statistic"] = " & ".join(s["statistic"] for s in imgidx_stats)  # type: ignore

        stat, nearest = stat_dict["statistic"], stat_dict["nearest"]
        kwargs_perimg.append(dict(linewidth=1, alpha=1, label=f"{stat} (AUPImO={nearest:.1%}) (imgidx={imgidx})"))

    fig, ax = plot_pimo_curves(
        shared_fpr,
        tprs,
        image_classes,
        *kwargs_perimg,
        # ---
        ax=ax,
        logfpr=logfpr,
        logfpr_epsilon=logfpr_epsilon,
        **kwargs_shared,
    )

    def _sort_pimo_of_boxplot_legend(handles: list, labels: list[str]):
        """sort the legend by label and put 'flier' at the bottom
        not essential but it makes the legend 'more deterministic' and organized
        """

        # [(handle0, label0), (handle1, label1),...]
        handles_labels = list(zip(handles, labels))
        handles_labels = sorted(handles_labels, key=lambda tup: tup[1])

        # ([handle0, handle1, ...], [label0, label1, ...])
        handles, labels = tuple(map(list, zip(*handles_labels)))  # type: ignore

        # put flier at the last position
        if "flier" in labels:
            idx = labels.index("flier")
            handles.append(handles.pop(idx))
            labels.append(labels.pop(idx))

        return handles, labels

    ax.legend(
        *_sort_pimo_of_boxplot_legend(*ax.get_legend_handles_labels()),
        title="boxplot stats",
        loc="lower right",
        fontsize="small",
        title_fontsize="small",
    )

    ax.set_title("Per-Image Overlap Curves (only AUC boxplot statistics)")

    return fig, ax


def _plot_aupimo_boxplot(aucs: Tensor, image_classes: Tensor, ax: Axes | None = None) -> tuple[Figure | None, Axes]:
    _validate_aucs(aucs, nan_allowed=True)
    _validate_atleast_one_anomalous_image(image_classes)

    fig, ax = _plot_perimg_metric_boxplot(
        values=aucs,
        image_classes=image_classes,
        only_class=1,
        ax=ax,
    )

    # don't go beyond the [0, 1] -+ \epsilon range in the X-axis
    XLIM_EPSILON = 0.01
    _bounded_lims(ax, axis=0, bounds=(0 - XLIM_EPSILON, 1 + XLIM_EPSILON))
    ax.xaxis.set_major_formatter(PercentFormatter(1))

    ax.set_xlabel("AUPImO [%]")
    ax.set_title("Area Under the Per-Image Overlap (AUPImO) Boxplot")

    return fig, ax


# =========================================== METRICS ===========================================

PImOResult = namedtuple(
    "PImOResult",
    [
        "thresholds",
        "fprs",
        "shared_fpr",
        "tprs",
        "image_classes",
    ],
)
PImOResult.__doc__ = """PImO result (from `PImO.compute()`).

[0] thresholds: shape (num_thresholds,), a `float` dtype as given in update()
[1] fprs: shape (num_images, num_thresholds), dtype `float64`, \in [0, 1]
[2] shared_fpr: shape (num_thresholds,), dtype `float64`, \in [0, 1]
[3] tprs: shape (num_images, num_thresholds), dtype `float64`, \in [0, 1] for anom images, `nan` for norm images
[4] image_classes: shape (num_images,), dtype `int32`, \in {0, 1}

- `num_thresholds` is an attribute of `PImO` and is given in the constructor (from parent class).
- `num_images` depends on the data seen by the model at the update() calls.
"""


class PImO(PerImageBinClfCurve):
    """Per-Image Overlap (PIMO, pronounced pee-mo) curve.

    PImO a measure of TP level across multiple thresholds,
        which are indexed by an FP measure on the normal images.

    At a given threshold:
        X-axis: False Positive metric shared across images:
            1. In-image FPR average on normal images (equivalent to the set FPR of normal images).
        Y-axis: Overlap between the class 'anomalous' in the ground truth and the predicted masks (in-image TPR).

    Note about other shared FPR alternatives:
        It can be made harder by using the cross-image max (or high-percentile) FPRs instead of the mean.
        I.e. the shared-fp axis (x-axies) is a statistic (across normal images) at each threshold.
        Rationale: this will further punish models that have exceptional FPs in normal images.
        Rationale: this will further punish models that have exceptional FPs in normal images.

    FP: False Positive
    FPR: False Positive Rate
    TP: True Positive
    TPR: True Positive Rate
    """

    def compute(self) -> PImOResult:  # type: ignore
        """Compute the PImO curve.

        Returns: PImOResult
        See `anomalib.utils.metrics.perimg.pimo.PImOResult` for details.
        """
        if self.is_empty:
            return PImOResult(
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.int32),
            )

        thresholds, binclf_curves, image_classes = super().compute()

        _validate_atleast_one_anomalous_image(image_classes)  # necessary for the TPR
        _validate_atleast_one_normal_image(image_classes)  # necessary for the shared FPR

        # (num_images, num_thresholds); from the parent class
        # fprs can be `nan` if an anomalous image is fully covered by the mask
        # but it's ok because we will use only the normal images
        tprs = PerImageBinClfCurve.tprs(binclf_curves)
        fprs = PerImageBinClfCurve.fprs(binclf_curves)

        # see note about shared FPR alternatives in the class's docstring
        shared_fpr = fprs[image_classes == 0].mean(dim=0)  # shape: (num_thresholds,)

        return PImOResult(thresholds, fprs, shared_fpr, tprs, image_classes)

    def plot(
        self,
        logfpr: bool = False,
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Plot shared FPR vs Per-Image Overlap (PImO) curves.

        Args:
            logfpr: whether to use log scale for the FPR axis (X-axis)

        Returns:
            fig, ax
        """
        if self.is_empty:
            raise RuntimeError("No data to plot.")

        _, __, shared_fpr, tprs, image_classes = self.compute()
        fig, ax = plot_pimo_curves(
            shared_fpr=shared_fpr,
            tprs=tprs,
            image_classes=image_classes,
            ax=ax,
            logfpr=logfpr,
        )
        ax.set_xlabel("Mean FPR on Normal Images")
        return fig, ax


class AUPImO(PImO):
    """Area Under the Per-Image Overlap (PImO) curve.

    AU is computed by the trapezoidal rule, each curve being treated separately.
    """

    def __init__(
        self,
        num_thresholds: int = 10_000,
        fpr_auc_ubound: float | Tensor = 1.0,
        **kwargs,
    ) -> None:
        """Area Under the Per-Image Overlap (PImO) curve.

        Args:
            num_thresholds: number of thresholds to use for the binclf curves
                            refer to `anomalib.utils.metrics.perimg.binclf_curve.PerImageBinClfCurve`
            fpr_auc_ubound: upper bound of the FPR range to compute the AUC

        """
        super().__init__(num_thresholds=num_thresholds, **kwargs)

        _validate_fpr_upper_bound(fpr_auc_ubound)
        self.register_buffer("fpr_auc_ubound", torch.as_tensor(fpr_auc_ubound, dtype=torch.float64))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fpr_auc_ubound={self.fpr_auc_ubound})"

    def compute(self) -> tuple[PImOResult, Tensor]:  # type: ignore
        """Compute the Area Under the Per-Image Overlap curves (AUPImO).

        Returns: (PImOResult, aucs)
            [0] PImOResult: PImOResult, see `anomalib.utils.metrics.perimg.pimo.PImOResult` for details.
            [1] aucs: shape (num_images,), dtype `float64`, \in [0, 1]
        """

        if self.is_empty:
            return PImOResult(
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.int32),
            ), torch.empty(0, dtype=torch.float64)

        pimoresult = thresholds, fprs, shared_fpr, tprs, image_classes = super().compute()

        # get the index of the value in `shared_fpr` that is closest to `self.fpr_auc_ubound in abs value
        # knwon issue: `shared_fpr[ubound_idx]` might not be exactly `self.fpr_auc_ubound`
        # but it's ok because `num_thresholds` should be large enough so that the error is negligible
        ubound_idx = torch.argmin(torch.abs(shared_fpr - self.fpr_auc_ubound))

        # limit the curves to the integration range [0, fpr_auc_ubound]
        # `shared_fpr` and `tprs` are in descending order; `flip()` reverts to ascending order
        tprs_auc: Tensor = tprs[:, ubound_idx:].flip(dims=(1,))
        shared_fpr_auc: Tensor = shared_fpr[ubound_idx:].flip(dims=(0,))

        aucs: Tensor = torch.trapezoid(tprs_auc, x=shared_fpr_auc, dim=1)

        # normalize the size of `aucs` by dividing by the x-range size
        aucs /= self.fpr_auc_ubound

        return pimoresult, aucs

    def plot_auc_boundary_conditions(
        self,
        axes: ndarray | None = None,
    ) -> tuple[Figure | None, ndarray]:
        """Plot the AUC boundary conditions based on FPR metrics on normal images."""

        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), width_ratios=[6, 8])
            fig.suptitle("AUPImO Integration Boundary Conditions")
            fig.set_tight_layout(True)
        elif not isinstance(axes, ndarray):
            raise ValueError(f"Expected argument `axes` to be an ndarray of matplotlib Axes, but got {type(axes)}.")
        elif axes.size != 2:
            raise ValueError(f"Expected argument `axes` to be of size 2, but got size {axes.size}.")
        else:
            fig, axes = (None, axes)

        axes = axes.flatten()

        (thresholds, fprs, shared_fpr, tprs, image_classes), aucs = self.compute()
        fprs_norm = fprs[image_classes == 0]

        # FRP upper bound is threshold lower bound
        thidx_lbound = torch.argmin(torch.abs(shared_fpr - self.fpr_auc_ubound))
        thbounds = (thresholds[thidx_lbound], thresholds[-1])

        ax = axes[1]
        ax.plot(thresholds, fprs_norm.T, alpha=0.3, color="gray", linewidth=0.5)
        ax.plot(thresholds, shared_fpr, color="black", linewidth=2, linestyle="--", label="mean")
        ax.axhline(
            self.fpr_auc_ubound,
            label=f"Shared FPR upper bound ({float(100 * self.fpr_auc_ubound):.2g}%)",
            linestyle="--",
            linewidth=1,
            color="red",
        )
        ax.axvline(
            thbounds[0],
            label="Threshold lower bound (@ FPR upper bound)",
            linestyle="--",
            linewidth=1,
            color="blue",
        )
        ax.add_patch(
            Rectangle(
                (thbounds[0], 0),
                thbounds[1] - thbounds[0],
                self.fpr_auc_ubound,
                facecolor="cyan",
                alpha=0.2,
                label="Integration range",
            )
        )

        ax.set_xlim(thresholds[0], thresholds[-1])
        ax.set_xlabel("Thresholds")

        YLIM_EPSILON = 0.01
        ax.set_ylabel("False Positive Rate (FPR)")
        YLIM_EPSILON = 0.01
        ax.set_ylim(0 - YLIM_EPSILON, 1 + YLIM_EPSILON)
        ax.yaxis.set_major_locator(FixedLocator(np.linspace(0, 1, 6)))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_minor_locator(FixedLocator(np.linspace(0, 1, 11)))

        ax.set_title("Thresholds vs FPR on Normal Images")
        ax.legend(loc="upper right", fontsize="small", title_fontsize="small")

        ax = axes[0]
        plot_fprs_vs_shared_fpr_predefviz(fprs, shared_fpr, image_classes, mode="norm-only", ax=ax)
        current_legend = ax.get_legend()
        handles = [
            ax.axvline(
                self.fpr_auc_ubound,
                label=f"Shared FPR upper bound ({float(100 * self.fpr_auc_ubound):.2g}%)",
                linestyle="--",
                linewidth=1,
                color="black",
            ),
            ax.axvspan(
                0,
                self.fpr_auc_ubound,
                label="Integration range",
                color="cyan",
                alpha=0.2,
            ),
        ]
        ax.legend(
            handles,
            [h.get_label() for h in handles],
            title="FPR AUC integration",
            loc="upper left",
            fontsize="small",
            title_fontsize="small",
        )
        ax.add_artist(current_legend)

        return fig, axes

    def plot_pimo_curves(
        self,
        logfpr: bool = False,
        show: str = "boxplot",
        integration_range: bool = True,
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes] | tuple[None, None]:
        """Plot shared FPR vs Per-Image Overlap (PImO) curves."""
        if self.is_empty:
            return None, None

        (thresholds, fprs, shared_fpr, tprs, image_classes), aucs = self.compute()

        if show == "all":
            fig, ax = plot_pimo_curves(
                shared_fpr,
                tprs,
                image_classes,
                ax=ax,
                logfpr=logfpr,
            )

        elif show == "boxplot":
            fig, ax = plot_pimo_curves_of_boxplot_stats(
                shared_fpr,
                tprs,
                image_classes,
                # ---
                self.boxplot_stats(),
                # ---
                ax=ax,
                logfpr=logfpr,
            )

        else:
            raise ValueError(f"Expected argument `show` to be one of 'all' or 'boxplot', but got {show}.")

        ax.set_xlabel("Mean FPR on Normal Images")

        if self.fpr_auc_ubound < 1 and integration_range:
            current_legend = ax.get_legend()
            handles = [
                ax.axvline(
                    self.fpr_auc_ubound,
                    label=f"upper bound ({float(100 * self.fpr_auc_ubound):.2g}%)",
                    linestyle="--",
                    linewidth=1,
                    color="black",
                ),
                ax.axvspan(
                    0,
                    self.fpr_auc_ubound,
                    label="interval",
                    color="cyan",
                    alpha=0.2,
                ),
            ]
            ax.legend(
                handles,
                [h.get_label() for h in handles],
                title="FPR AUC integration",
                loc="center right",
                fontsize="small",
                title_fontsize="small",
            )
            ax.add_artist(current_legend)

        return fig, ax

    def boxplot_stats(self) -> list[dict[str, str | int | float | None]]:
        """Compute boxplot stats of AUPImO values (e.g. median, mean, quartiles, etc.).

        Returns:
            list[dict[str, str | int | float | None]]: List of AUCs statistics from a boxplot.
            refer to `anomalib.utils.metrics.perimg.common._perimg_boxplot_stats()` for the keys and values.
        """
        (_, __, ___, ____, image_classes), aucs = self.compute()
        stats = _perimg_boxplot_stats(values=aucs, image_classes=image_classes, only_class=1)
        return stats

    def plot_boxplot(
        self,
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Plot boxplot of AUPImO values."""
        (thresholds, fprs, shared_fpr, tprs, image_classes), aucs = self.compute()
        fig, ax = _plot_aupimo_boxplot(
            aucs=aucs,
            image_classes=image_classes,
            ax=ax,
        )
        return fig, ax

    def plot(
        self,
        logfpr: bool = False,
        axes: Axes | ndarray | None = None,
    ) -> tuple[Figure | None, Axes | ndarray]:
        """Plot AUPImO boxplot with its statistics' PImO curves."""

        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), width_ratios=[6, 8])
            fig.suptitle("Area Under the Per-Image Overlap (AUPImO) Curves")
            fig.set_tight_layout(True)
        else:
            fig, axes = (None, axes)

        if isinstance(axes, Axes):
            return self.plot_pimo_curves(ax=axes, logfpr=logfpr, show="boxplot")

        if not isinstance(axes, ndarray):
            raise ValueError(f"Expected argument `axes` to be a matplotlib Axes or ndarray, but got {type(axes)}.")

        if axes.size != 2:
            raise ValueError(
                f"Expected argument `axes` , when type `ndarray`, to be of size 2, but got size {axes.size}."
            )

        axes = axes.flatten()
        self.plot_boxplot(ax=axes[0])
        axes[0].set_title("Boxplot")
        self.plot_pimo_curves(ax=axes[1], logfpr=logfpr, show="boxplot")
        axes[1].set_title("PImO Curves")
        return fig, axes


class AULogPImO(PImO):
    """Area Under the Per-Image Overlap (PIMO, pronounced pee-mo) curves with log(FPR) (instead of FPR) in the X-axis

    This will further give more importance/visibility to the low FPR region.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("**coming up later**")
