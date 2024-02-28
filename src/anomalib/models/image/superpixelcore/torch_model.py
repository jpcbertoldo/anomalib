from collections.abc import Sequence

import numpy as np
import torch
from aupimo.oracles_numpy import (
    get_superpixels_watershed,
)
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.models.components import TimmFeatureExtractor


class SuperpixelCoreModel(nn.Module):
    """SuperpixelCore Module.

    Args:
        input_size (tuple[int, int]): Input size for the model.
        layers (list[str]): Layers used for feature extraction
        backbone (str, optional): Pre-trained model backbone.
            Defaults to ``resnet18``.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
            Defaults to ``True``.
        num_neighbors (int, optional): Number of nearest neighbors.
            Defaults to ``9``.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        layers: Sequence[str],
        backbone: str,
        superpixel_relsize: float,
        num_neighbors: int = 9,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.layers = layers
        self.input_size = input_size
        self.num_neighbors = num_neighbors

        self.superpixel_relsize = superpixel_relsize

        self.feature_extractor = TimmFeatureExtractor(
            backbone=self.backbone,
            pre_trained=True,
            layers=self.layers,
        )

        self.register_buffer("memory_bank", torch.Tensor())
        self.memory_bank: torch.Tensor

    def forward(
        self,
        input_tensor: torch.Tensor,
        image_original: torch.Tensor,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Args:
        input_tensor (torch.Tensor): B, C, H, W
        image_original (torch.Tensor): B, W, H, C
        """
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        features = {
            layer: F.interpolate(
                feature,
                size=self.input_size,
                mode="bilinear",
            )
            for layer, feature in features.items()
        }
        embeddings = torch.cat([features[layer] for layer in self.layers])

        superpixels = torch.from_numpy(
            np.stack(
                [
                    get_superpixels_watershed(
                        # img.permute(2, 1, 0).numpy(),
                        img.numpy(),
                        superpixel_relsize=self.superpixel_relsize,
                        compactness=1e-4,
                    )
                    for img in image_original
                ],
            ),
        )

        # list[list[int]]: list of superpixels labels for each image
        superpixels_labels = {tuple(sorted(set(suppix.flatten().tolist()))) for suppix in superpixels}
        assert len(superpixels_labels) == 1, f"{len(superpixels_labels)=}"
        assert (superpixels_labels := next(iter(superpixels_labels))) == tuple(
            list(range(1, len(superpixels_labels) + 1)),
        )

        # superpixels: [B, H, W], embeddings: [B, C, H, W]
        # ==>  (superpixels) embeddings: [B, C, num_suppix]
        embeddings = torch.stack(
            [
                # [C, num_suppix]
                torch.stack(
                    [
                        # emb: [C, H, W], suppix: [H, W]  ==>  [C,]
                        emb[:, (suppix == label)].mean(dim=1)
                        for label in superpixels_labels
                    ],
                    dim=1,
                )
                for emb, suppix in zip(embeddings, superpixels, strict=True)
            ],
            dim=0,
        )

        # embedding: [B, C, num_suppix]  ==>  [B * num_suppix, C]
        batch_size, embedding_size, num_suppix = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1).reshape(-1, embedding_size)

        if self.training:
            return {"embeddings": embeddings, "superpixels": superpixels}

        # superpixel_scores & locations: [B * num_suppix, 1]
        superpixel_scores, locations = self.nearest_neighbors(embedding=embeddings, n_neighbors=1)
        # reshape to [B, num_suppix]
        superpixel_scores = superpixel_scores.reshape((batch_size, -1))
        locations = locations.reshape((batch_size, -1))
        # compute anomaly score
        pred_score = self.compute_anomaly_score(superpixel_scores, locations, embeddings)
        # pred_score = superpixel_scores.amax(1)
        # reshape to w, h
        # get anomaly map
        anomaly_map = torch.zeros_like(superpixels, dtype=superpixel_scores.dtype)
        # for loop on the batch dim bc not all superpixels have the same number of pixels
        for image_idx in range(batch_size):
            for label in superpixels_labels:
                anomaly_map[image_idx][superpixels[image_idx] == label] = superpixel_scores[image_idx, label - 1]
                # -1 because labels start at 1
        return {"anomaly_map": anomaly_map, "pred_score": pred_score}

    @staticmethod
    def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)  # |x|
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)  # |y|
        # row distance can be rewritten as sqrt(|x| - 2 * x @ y.T + |y|.T)
        res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
        return res.clamp_min_(0).sqrt_()

    def nearest_neighbors(self, embedding: torch.Tensor, n_neighbors: int) -> tuple[torch.Tensor, torch.Tensor]:
        distances = self.euclidean_dist(embedding, self.memory_bank)
        if n_neighbors == 1:
            # when n_neighbors is 1, speed up computation by using min instead of topk
            superpixel_scores, locations = distances.min(1)
        else:
            superpixel_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
        return superpixel_scores, locations

    def compute_anomaly_score(
        self,
        patch_scores: torch.Tensor,
        locations: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        # raise NotImplementedError
        # Don't need to compute weights if num_neighbors is 1
        if self.num_neighbors == 1:
            return patch_scores.amax(1)
        batch_size, num_patches = patch_scores.shape
        # 1. Find the patch with the largest distance to it's nearest neighbor in each image
        max_patches = torch.argmax(patch_scores, dim=1)  # indices of m^test,* in the paper
        # m^test,* in the paper
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]
        # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
        score = patch_scores[torch.arange(batch_size), max_patches]  # s^* in the paper
        nn_index = locations[torch.arange(batch_size), max_patches]  # indices of m^* in the paper
        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = self.memory_bank[nn_index, :]  # m^* in the paper
        # indices of N_b(m^*) in the paper
        memory_bank_effective_size = self.memory_bank.shape[0]  # edge case when memory bank is too small
        _, support_samples = self.nearest_neighbors(
            nn_sample,
            n_neighbors=min(self.num_neighbors, memory_bank_effective_size),
        )
        # 4. Find the distance of the patch features to each of the support samples
        distances = self.euclidean_dist(max_patches_features.unsqueeze(1), self.memory_bank[support_samples])
        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        # 6. Apply the weight factor to the score
        return weights * score  # s in the paper
