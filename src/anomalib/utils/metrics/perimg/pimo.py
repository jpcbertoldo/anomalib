


from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.pyplot import Figure
from numpy import ndarray
from torch import Tensor

from anomalib.data.utils.image import duplicate_filename

from .binclf_curve import PerImageBinClfCurve
from .common import (
    _validate_and_convert_aucs,
    _validate_and_convert_fpath,
    _validate_and_convert_rate,
    _validate_image_classes,
    _validate_per_image_rate_curves,
    _validate_rate_curve,
    _validate_threshs,
    per_image_boxplot_stats,
)
from .plot import (
    _add_avline_at_score_random_model,
    _add_integration_range_to_piMo_curves,
    _format_axis_rate_metric_log,
    plot_all_piMo_curves,
    plot_aupiMo_boxplot,
    plot_boxplot_logpiMo_curves,
    plot_boxplot_piMo_curves,
    plot_pimfpr_curves_norm_only,
    plot_th_fpr_curves_norm_only,
)


# =========================================== METRICS ===========================================

# TODO review where this is used (check where compute() is called) and use it as type hint
@dataclass
class PIMOResult:
    """PIMO result (from `PIMO.compute()`).

    The attribute `shared_fpr_metric` is a user-defined parameter of the curve.

    threshs: shape (num_threshs,), a `float` dtype as given in update()
    fprs: shape (num_images, num_threshs), dtype `float64`, \\in [0, 1]
    shared_fpr: shape (num_threshs,), dtype `float64`, \\in [0, 1]
    per_image_tprs: shape (num_images, num_threshs), dtype `float64`, \\in [0, 1] for anom images, `nan` for norm images
    image_classes: shape (num_images,), dtype `int32`, \\in {0, 1}

    - `num_threshs` comes from `PIMO` and is given in the constructor (from parent class).
    - `num_images` depends on the data seen by the model at the update() calls.
    """

    def plot(
        self,
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Plot shared FPR vs Per-Image Overlap (PIMO) curves."""

        if self.is_empty:
            return None, None

        _, __, shared_fpr, per_image_tprs, image_classes = self.compute().to_tuple()

        fig, ax = plot_all_piMo_curves(
            shared_fpr,
            per_image_tprs,
            image_classes,
            ax=ax,
        )
        # `MEAN_PERIMAGE_FPR` is the only one implemented for now
        ax.set_xlabel("Mean FPR on Normal Images")

        return fig, ax

@dataclass
class AUPIMOResult:
    """Area Under the Per-Image Overlap (PIMO) curve.

    The attributes `shared_fpr_metric`, `lbound`, and `ubound` are
    user-defined parameters of the metric.

    aucs: shape (num_images,), dtype `float64`, \\in [0, 1] for anom images, `nan` for norm images
    """

    def plot_all_logpiMo_curves(
        self,
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Plot log10( shared FPR ) vs Per-Image Overlap (LogPIMO) curves (all curves)."""

        if self.is_empty:
            return None, None

        curves, _ = self.compute()
        _, __, shared_fpr, per_image_tprs, image_classes = curves.to_tuple()

        fig, ax = plot_all_piMo_curves(
            shared_fpr,
            per_image_tprs,
            image_classes,
            ax=ax,
        )
        ax.set_xlabel("Log10 of Mean FPR on Normal Images")
        ax.set_title("Log Per-Image Overlap (LogPIMO) Curves")
        _format_axis_rate_metric_log(ax, axis=0, lower_lim=self.lbound, upper_lim=self.ubound)
        # they are not exactly the same as the input because the function above rounds them
        xtickmin, xtickmax = ax.xaxis.get_ticklocs()[[0, -1]]
        _add_integration_range_to_piMo_curves(
            ax, (self.lbound, self.ubound), span=(xtickmin < self.lbound or xtickmax > self.ubound)
        )

        return fig, ax

    def boxplot_stats(self) -> list[dict[str, str | int | float | None]]:
        """Compute boxplot stats of AUPIMO values (e.g. median, mean, quartiles, etc.).

        Returns:
            list[dict[str, str | int | float | None]]: List of AUCs statistics from a boxplot.
            refer to `anomalib.utils.metrics.per_image.common.per_image_boxplot_stats()` for the keys and values.
        """
        _, aupiMos = self.compute()
        stats = per_image_boxplot_stats(values=aupiMos.aucs, image_classes=aupiMos.image_classes, only_class=1)
        return stats

    def plot_boxplot_logpiMo_curves(
        self,
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Plot log10( shared FPR ) vs Per-Image Overlap (LogPIMO) curves (boxplot images only).
        The 'boxplot images' are those from the boxplot of AUPIMO values (see `AUPIMO.boxplot_stats()`).
        """

        if self.is_empty:
            return None, None

        curves, _ = self.compute()
        _, __, shared_fpr, per_image_tprs, image_classes = curves.to_tuple()
        fig, ax = plot_boxplot_logpiMo_curves(
            shared_fpr,
            per_image_tprs,
            image_classes,
            self.boxplot_stats(),
            self.lbound,
            self.ubound,
            ax=ax,
        )
        ax.set_xlabel("Log10 of Mean FPR on Normal Images")
        return fig, ax

    def plot_boxplot(
        self,
        ax: Axes | None = None,
    ) -> tuple[Figure | None, Axes]:
        """Plot boxplot of AUPIMO values."""

        if self.is_empty:
            return None, None

        _, aupiMos = self.compute()
        fig, ax = plot_aupiMo_boxplot(aupiMos.aucs, aupiMos.image_classes, self.random_model_auc, ax=ax)
        return fig, ax

    def plot(
        self,
        ax: Axes | ndarray | None = None,
    ) -> tuple[Figure | None, Axes | ndarray]:
        """Plot AUPIMO boxplot with its statistics' LogPIMO curves."""

        if self.is_empty:
            return None, None

        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(14, 6), width_ratios=[6, 8])
            fig.suptitle("Area Under the Log Per-Image Overlap (AUPIMO) Curves")
            fig.set_tight_layout(True)
        else:
            fig, ax = (None, ax)

        if isinstance(ax, Axes):
            return self.plot_boxplot_logpiMo_curves(ax=ax)

        if not isinstance(ax, ndarray):
            raise ValueError(f"Expected argument `ax` to be a matplotlib Axes or ndarray, but got {type(ax)}.")

        if ax.size != 2:
            raise ValueError(f"Expected argument `ax` , when type `ndarray`, to be of size 2, but got size {ax.size}.")

        ax = ax.flatten()
        self.plot_boxplot(ax=ax[0])
        self.plot_boxplot_logpiMo_curves(ax=ax[1])

        if fig is not None:  # it means the ax were created by this function (and so was the suptitle)
            ax[0].set_title("AUC Boxplot")
            ax[1].set_title("Curves")

        return fig, ax

    def plot_per_image_fprs(
        self,
        ax: ndarray | None = None,
    ) -> tuple[Figure | None, ndarray]:
        """Plot the AUC boundary conditions based on log(FPR) on normal images.
        Args:
            ax: ndarray of matplotlib Axes of size 2, or None.
                If None, the function will create the ax.
        Returns:
            tuple[Figure | None, ndarray]: (fig, ax)
                fig: matplotlib Figure
                ax: ndarray of matplotlib Axes of size 2
        """

        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(14, 6), width_ratios=[6, 8])
            fig.suptitle("AUPIMO Integration Boundary Conditions")
            fig.set_tight_layout(True)
        elif not isinstance(ax, ndarray):
            raise ValueError(f"Expected argument `ax` to be an ndarray of matplotlib Axes, but got {type(ax)}.")
        elif ax.size != 2:
            raise ValueError(f"Expected argument `ax` to be of size 2, but got size {ax.size}.")
        else:
            fig, ax = (None, ax)

        ax = ax.flatten()

        curves, __ = self.compute()
        threshs, fprs, shared_fpr, _, image_classes = curves.to_tuple()

        # FRP upper bound is thresh lower bound
        th_lbound = curves.thresh_at(self.ubound)

        # FPR lower bound is thresh upper bound
        th_ubound = curves.thresh_at(self.lbound)

        plot_th_fpr_curves_norm_only(
            fprs,
            shared_fpr,
            threshs,
            image_classes,
            th_lb_fpr_ub=(th_lbound, self.ubound),
            th_ub_fpr_lb=(th_ubound, self.lbound),
            ax=ax[0],
        )

        plot_pimfpr_curves_norm_only(fprs, shared_fpr, image_classes, ax=ax[1])
        _add_integration_range_to_piMo_curves(ax[1], (self.lbound, self.ubound))

        return fig, ax

# TODO(jpcbertoldo): warn when the rations from binclf are too imprecise  # noqa: TD003
    # image_classes: list[Tensor]
    # self.image_classes.append(
    #     # an image is anomalous if it has at least one anomaly pixel
    #     (masks.flatten(1) == 1).any(dim=1).to(torch.int32),
    # )
