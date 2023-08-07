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

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
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
    

def _validate_kwargs_perimg(kwargs_perimg: list[dict[str, Any] | None], num_images: int) -> None:
    
    if len(kwargs_perimg) == 0:
        pass

    elif len(kwargs_perimg) != num_images:
        raise ValueError(
            f"Expected argument `kwargs_perimg` to have the same number of dicts as number of images, "
            f"but got {len(kwargs_perimg)} dicts while {num_images} images."
        )

    elif len(othertypes := {type(kws) for kws in kwargs_perimg if kws is not None and not isinstance(kws, dict)}) == 0:
        raise ValueError(
            "Expected argument `kwargs_perimg` to be a list of dicts or Nones, "
            f"but found {sorted(othertypes, key=lambda t: t.__name__)} instead."
        )


# =========================================== PLOT ===========================================


def plot_pimo_curves(
    shared_fpr: Tensor,
    tprs: Tensor,
    image_classes: Tensor,
    *kwargs_perimg: dict[str, Any] | None,
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
    if image_classes is not None:
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

    imgidxs_toplot_fliers = {s["imgidx"] for s in aupimo_boxplot_stats if s["statistic"] in ("flierlo", "flierhi")}
    # it is sorted so that only the first one has a label (others are plotted but don't show in the legend)
    imgidxs_toplot_fliers = sorted(imgidxs_toplot_fliers)
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
        title="boxplot stats", loc="lower right", fontsize="small", title_fontsize="small"
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

    def compute(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:  # type: ignore
        """Compute the PImO curve.


        Returns: (thresholds, shared_fpr, tprs, image_classes)
            [0] thresholds: shape (num_thresholds,), dtype as given in update()
            [1] shared_fpr: shape (num_thresholds,), dtype float64, \in [0, 1]
            [2] tprs: shape (num_images, num_thresholds), dtype float64,
                        \in [0, 1] for anomalous images, `nan` for normal images
            [3] image_classes: shape (num_images,), dtype int32, \in {0, 1}

                `num_thresholds` is an attribute of the parent class.
                `num_images` depends on the data seen by the model at the update() calls.
        """
        if self.is_empty:
            return (
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.int64),
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

        return thresholds, shared_fpr, tprs, image_classes

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

        _, shared_fpr, tprs, image_classes = self.compute()
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

    TODO get lower bound from shared fpr and only compute the area under the curve in the considered region
        --> (1) add class attribute, (2) add integration range at plot, (3) filter curves before intergration
    """

    def compute(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:  # type: ignore
        """Compute the Area Under the Per-Image Overlap curves (AUPImO).

        Returns: (thresholds, shared_fpr, tprs, image_classes, aucs)
            [0] thresholds: shape (num_thresholds,), dtype as given in update()
            [1] shared_fpr: shape (num_thresholds,), dtype float64, \in [0, 1]
            [2] tprs: shape (num_images, num_thresholds), dtype float64,
                        \in [0, 1] for anomalous images, `nan` for normal images
            [3] image_classes: shape (num_images,), dtype int32, \in {0, 1}
            [4] aucs: shape (num_images,), dtype float64, \in [0, 1]
        """

        if self.is_empty:
            return (
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.float64),
                torch.empty(0, dtype=torch.int64),
                torch.empty(0, dtype=torch.float64),
            )

        thresholds, shared_fpr, tprs, image_classes = super().compute()

        # TODO find lower bound from shared fpr

        # `shared_fpr` and `tprs` are in descending order; `flip()` reverts to ascending order
        aucs: Tensor = torch.trapezoid(tprs.flip(dims=(1,)), x=shared_fpr.flip(dims=(0,)), dim=1)
        return thresholds, shared_fpr, tprs, image_classes, aucs

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

        thresholds, shared_fpr, tprs, image_classes, aucs = self.compute()

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
                ax.axvline(self.fpr_auc_ubound, label=f"FPR upper bound ({float(100 * self.fpr_auc_ubound):.2g}%)", linestyle="--", linewidth=1, color="black",),
                ax.axvspan(0, self.fpr_auc_ubound, label="interval", color='cyan', alpha=0.2,),
            ]
            ax.legend(handles, [h.get_label() for h in handles], title="AUC", loc="center right", fontsize="small", title_fontsize="small")
            ax.add_artist(current_legend)
    
        return fig, ax

    def boxplot_stats(self) -> list[dict[str, str | int | float | None]]:
        """Compute boxplot stats of AUPImO values (e.g. median, mean, quartiles, etc.).

        Returns:
            list[dict[str, str | int | float | None]]: List of AUCs statistics from a boxplot.
            refer to `anomalib.utils.metrics.perimg.common._perimg_boxplot_stats()` for the keys and values.
        """
        _, __, ___, image_classes, aucs = self.compute()

        stats = _perimg_boxplot_stats(values=aucs, image_classes=image_classes, only_class=1)
        return stats

    def plot_boxplot(
        self,
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Plot boxplot of AUPImO values."""
        thresholds, shared_fpr, tprs, image_classes, aucs = self.compute()
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
        super().__init__(*args, **kwargs)
        raise NotImplementedError("**coming up later**")
