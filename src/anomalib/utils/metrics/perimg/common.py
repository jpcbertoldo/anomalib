"""Common utilities (e.g. validations) and x-metric functionalities (boxplot statistics, statistical comparisons).

Boxplot statistics

    For a single per-image metric collection (1 model, 1 dataset), compute statistics and find the closest image
    to each statistic.

Statistical tests

    For two or more per-image metric collections (2+ models, 1 dataset), compare all pairs of models using a
    parametric or non-parametric test over the paired per-image metric values.

    Parametric test: paired t-test.

        Refs:
            - `scipy.stats.ttest_rel`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
            - Wikipedia page: https://en.wikipedia.org/wiki/Student's_t-test#Dependent_t-test_for_paired_samples

    Non-parametric test: Wilcoxon signed rank test.
            - `scipy.stats.wilcoxon`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html#scipy.stats.wilcoxon
            - Wikipedia page: https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas
import scipy.stats
import torch
from matplotlib import cm
from numpy import ndarray

# =========================================== ARGS VALIDATION ===========================================




# =========================================== FUNCTIONAL ===========================================





    # confidences = {(model1, model2): 1 - tr.pvalue for (model1, model2), tr in test_results.items()}
    # confidences.update({(i, i): np.nan for i in sorted_models})

    # df = pandas.DataFrame(confidences, index=["confidence"]).T
    # df.index.names = ["model1", "model2"]
    # df = df.pivot_table(index="model1", columns="model2", values="confidence", dropna=False)
    # df = df[sorted_models[1:]]  # sort columns; [1:] because the first column is empty
    # df = df.T[sorted_models].T  # sort rows

    # df["Average"] = np.array(sorted_models_averages)
    # df = df.set_index("Average", append=True)

    # cmap = cm.inferno
    # cmap.set_bad("black")

    # def fmt(x):
    #     if np.isnan(x):
    #         return "."
    #     return f"{x:.1%}"

    # confidence_table = df.style.format(fmt).background_gradient(cmap=cmap, vmin=0, vmax=1)

    # return confidence_table




def compare_models_pairwise_wilcoxon(
    models: dict[str, ndarray],
    higher_is_better: bool = True,
    return_test_results: bool = False,
    atol: float | None = 0.001,
):
    """Compare all pairs of models using a non-parametric test (wilcoxon signed rank test).

    Models are sorted by average rank (`atol` ignored), then compared pairwise assuming that
    the first model is better than the second model, and better than third one, and second is better than third one ...

    Each comparison of two models is a [paired] wilcoxon signed rank test with the alternative hypothesis that
    the first model is better than the second model (null hypothesis is that they are equal).

    Args:
        models (dict[str, ndarray]): Dictionary of models and the per-image values of the metric.
        higher_is_better (bool): Whether higher values of the metric are better. Defaults to True.
        return_test_results (bool):
            `True`: (sorted_models, test_results)
                sorted_models: list of model names sorted by average rank
                test_results: dict of (model1, model2) -> wilcoxon result (from scipy)
            `False`: confidence_table
                confidence_table: pandas DataFrame of confidence that model1 > model2 (higher means more confident)
    """

    # ** validate **
    _validate_scores_per_model(models)

    if atol is not None:
        atol = float(_validate_and_convert_rate(atol, nonzero=True, nonone=False))

    # ** compute **

    # index is not the image index! because the `nan`s were removed
    df = pandas.DataFrame(models)[models_sorted_abc]


    avgrank_permodel = dict(zip(models_sorted_abc, models_avgranks_abc))

    # sort models by average value
    avgrank_permodel_sorted = sorted(avgrank_permodel.items(), key=lambda kv: kv[1], reverse=False)

    # model[0] > model[1], model[0] > model[2], model[1] > model[2], ...
    num_models = len(models)
    comparisons = list(itertools.combinations(range(num_models), 2))

    # for each comparison, compute the confidence (1 - p-value) that model[i] > model[j]
    test_results = {}

    # `i` and `j` are indices of the sorted models
    for i, j in comparisons:
        # _ is the average rank
        (model_i, _), (model_j, _) = avgrank_permodel_sorted[i], avgrank_permodel_sorted[j]
        model_i_values = models[model_i]
        model_j_values = models[model_j]

        diff = model_i_values - model_j_values

        if atol is not None:
            # make the difference null if below the tolerance
            diff[diff.abs() <= atol] = 0.0

        # extreme case
        if (diff == 0).all():
            test_results[(model_i, model_j)] = scipy.stats._morestats.WilcoxonResult(np.nan, 1.0)
            continue

        # assume `model_i` is greater than `model_j` (assume less if `higher_is_better=False``)
        test_results[(model_i, model_j)] = scipy.stats.wilcoxon(
            diff,
            alternative="greater" if higher_is_better else "less",
        )

    sorted_models = [m for m, _ in avgrank_permodel_sorted]

    return sorted_models, test_results

    confidences = {(model1, model2): 1 - tr.pvalue for (model1, model2), tr in test_results.items()}
    confidences.update({(i, i): np.nan for i in sorted_models})

    df = pandas.DataFrame(confidences, index=["confidence"]).T
    df.index.names = ["model1", "model2"]
    df = df.pivot_table(index="model1", columns="model2", values="confidence", dropna=False)
    df = df[sorted_models[1:]]  # sort columns; [1:] because the first column is empty
    df = df.T[sorted_models].T  # sort rows

    df["Average Rank"] = np.array(val for _, val in avgrank_permodel_sorted)
    df = df.set_index("Average Rank", append=True)

    cmap = cm.inferno
    cmap.set_bad("black")

    def fmt(x):
        if np.isnan(x):
            return "."
        return f"{x:.1%}"

    confidence_table = df.style.format(fmt).background_gradient(cmap=cmap, vmin=0, vmax=1)

    return confidence_table
