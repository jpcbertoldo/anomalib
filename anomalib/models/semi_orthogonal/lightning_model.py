"""PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization.

Paper https://arxiv.org/abs/2011.08785
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch import Tensor, nn

from anomalib.models.components import AnomalyModule, FeatureExtractor

logger = logging.getLogger(__name__)

__all__ = ["SemiOrthogonal", "SemiOrthogonalLightning"]


class SemiOrthogonalMahalanobisEvaluator(nn.Module):
    def __init__(
        self,
        cov: Tensor,
        mean: Tensor,
        embedding_map: Tensor,
        # k=100, method='ortho', num_samples=None,
        # eps = 1e-2 is the default in PaDiM (Defard et al., 2021)
        eps: float = 1e-2,
    ):
        super(SemiOrthogonalMahalanobisEvaluator, self).__init__()
        self.cov: Tensor = cov  # hwkk
        self.mean: Tensor = mean  # hwk
        self.eps: float = eps  # 1e-2 from Defard et al. (2021)
        # self.method = method
        # self.num_samples = num_samples
        # self.k = k
        # h, w, c, d = cov.size()
        precision_matrix: Tensor = self.build()  # hwmk
        self.register_buffer(
            "precision_matrix", precision_matrix
        )  # todo this should be a buffer with none at first, then the module should set it?
        self.register_buffer("embedding_map", embedding_map)

    def forward(self, x):
        m = self.mean.transpose(2, 1).transpose(1, 0).unsqueeze(0)  # hwk -> hkw -> khw -> 1khw
        # n, c, h, w
        # batch = x[0]  # idk why [0] so far
        batch = x  # idk why [0] so far
        batch_reduced = torch.einsum("nchw, ck -> nkhw", batch, self.embedding_map)
        batch_reduced -= m  # nkhw
        anoma_scores_squared = torch.einsum(
            "nmhw, hwmk, nkhw -> nhw", batch_reduced, self.precision_matrix, batch_reduced
        ).unsqueeze(1)
        return anoma_scores_squared.abs().sqrt()

    def build(self) -> Tensor:
        print("build a precision matrix...")
        identity = torch.eye(self.cov.size(-1), device=self.cov.device).unsqueeze(0).unsqueeze(0)
        return (self.cov + self.eps * identity).inverse()

    @staticmethod
    def get_embedding(fin, fout) -> Tensor:
        W = torch.Tensor(fin, fout)
        nn.init.orthogonal_(W)
        return W


DIMS: Dict[str, Dict[str, Union[int, Dict[str, int]]]] = {
    "resnet18": {
        "orig_dims": 448,
        "reduced_dims": 100,
        "emb_scale": 4,
        "layers_spatial_size": {"layer1": 64, "layer2": 32, "layer3": 16},
    },
    "wide_resnet50_2": {"orig_dims": 1792, "reduced_dims": 300, "emb_scale": 4},
}


# todo good practices for torch module
# https://pytorch.org/docs/stable/notes/modules.html#a-simple-custom-module:~:text=Provide%20a%20device,for%20an%20explanation.


class SemiOrthogonalModel(nn.Module):
    def __init__(
        self,
        layers: List[str],
        input_size: Tuple[int, int],
        backbone: str,
        pre_trained: bool = True,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.input_size = input_size
        self.backbone = backbone
        self.pre_trained = pre_trained
        self.feature_extractor = FeatureExtractor(backbone=self.backbone, layers=layers, pre_trained=pre_trained)

        self.dims: dict = DIMS[backbone]
        self.orig_dims = self.dims["orig_dims"]
        self.reduced_dims = self.dims["reduced_dims"]
        self.emb_scale = self.dims["emb_scale"]
        self.largest_spatial_size = self.dims["layers_spatial_size"][self.layers[0]]

        # todo: make this controllable with the seed
        # c, d
        self.embedding_map = SemiOrthogonalMahalanobisEvaluator.get_embedding(self.orig_dims, self.reduced_dims)

        self.covariance_matrices = None
        self.means = None
        self.mahalanobis_evaluator: Optional[SemiOrthogonalMahalanobisEvaluator] = None

    def forward(self, x: Tensor) -> Tensor:
        self.feature_extractor.eval()
        features: Tensor = self.feature_extractor(x)

        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        if self.training:
            return embeddings

        anomaly_scores = self.mahalanobis_evaluator(embeddings)
        return F.interpolate(anomaly_scores, size=self.input_size[0], mode="nearest")


@MODEL_REGISTRY
class SemiOrthogonal(AnomalyModule):
    def __init__(
        self,
        layers: List[str],
        input_size: Tuple[int, int],
        backbone: str,
        pre_trained: bool = True,
    ):
        super().__init__()

        self.model: SemiOrthogonalModel = SemiOrthogonalModel(
            layers=layers,
            input_size=input_size,
            backbone=backbone,
            pre_trained=pre_trained,
        ).train()

        # accu = "accumulated"
        # i will keep the accumulated covariance and mean (without dividing by the number of samples)
        # because then the training can continue after the first time it passes by validation epoch

        # h, w, d, d
        self.covariance_matrices_accu = torch.zeros(
            self.model.largest_spatial_size,
            self.model.largest_spatial_size,
            self.model.reduced_dims,
            self.model.reduced_dims,
        )  # covariance

        # h, w, d
        self.means_accu = torch.zeros(
            self.model.largest_spatial_size,
            self.model.largest_spatial_size,
            self.model.reduced_dims,
        )  # mean

        logger.info(f"{self.covariance_matrices_accu.shape=}")
        logger.info(f"{self.means_accu.shape=}")
        logger.info(f"{self.model.embedding_map.shape=}")

        self.nsamples_seen = 0
        self.loss = None

    @staticmethod
    def configure_optimizers():  # pylint: disable=arguments-differ
        return None

    def training_step(self, batch, _batch_idx):  # pylint: disable=arguments-differ
        # model: (b, 3, h, w) -> (b, d, h, w)
        # already reduced!
        embeddings = self.model(batch["image"])

        # todo: encapsulate this in the semi orthogonal mahalanobis evaluator
        # b, d, h, w
        embeddings = torch.einsum("bchw, cd -> bdhw", (embeddings, self.model.embedding_map))

        # h, w, d, d
        self.covariance_matrices_accu += torch.einsum("bDhw, bdhw -> hwDd", (embeddings, embeddings))

        # h, w, d
        self.means_accu += embeddings.sum(0).transpose(0, 1).transpose(1, 2)  # bdhw -> dhw -> hdw -> hwd

        self.nsamples_seen += batch["image"].size(0)

    def on_validation_start(self) -> None:
        # NOTE: Previous anomalib versions fit Gaussian at the end of the epoch.
        #   This is not possible anymore with PyTorch Lightning v1.4.0 since validation
        #   is run within train epoch.
        # logger.info("Aggregating the embedding extracted from the training set.")
        # embeddings = torch.vstack(self.embeddings)
        # logger.info("Fitting a Gaussian to the embedding collected from the training set.")
        # self.stats = self.model.gaussian.fit(embeddings)

        self.model.eval()

        means = self.means_accu / self.nsamples_seen
        covariance_matrices = self.covariance_matrices_accu / self.nsamples_seen
        covariance_matrices -= torch.einsum("hwD, hwd -> hwDd", (means, means))  # unbiased

        # todo: encapsulate this in the model

        self.model.means = means
        self.model.covariance_matrices = covariance_matrices

        self.model.mahalanobis_evaluator = SemiOrthogonalMahalanobisEvaluator(
            covariance_matrices, means, self.model.embedding_map
        )

    def validation_step(self, batch, _):  # pylint: disable=arguments-differ
        batch["anomaly_maps"] = self.model(batch["image"])
        return batch


class SemiOrthogonalLightning(SemiOrthogonal):
    def __init__(self, hparams: Union[DictConfig, ListConfig]):
        super().__init__(
            input_size=hparams.model.input_size,
            layers=hparams.model.layers,
            backbone=hparams.model.backbone,
        )
        self.hparams: Union[DictConfig, ListConfig]  # type: ignore
        self.save_hyperparameters(hparams)
