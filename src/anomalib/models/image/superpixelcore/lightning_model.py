import logging
from collections.abc import Sequence
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib import LearningType
from anomalib.models.components import AnomalyModule, KCenterGreedy, MemoryBankMixin

from .torch_model import SuperpixelCoreModel

logger = logging.getLogger(__name__)


class SuperpixelCore(MemoryBankMixin, AnomalyModule):
    def __init__(
        self,
        input_size: tuple[int, int],
        backbone: str,
        layers: Sequence[str],
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
    ) -> None:
        super().__init__()
        self.model: SuperpixelCoreModel = SuperpixelCoreModel(
            input_size=input_size,
            backbone=backbone,
            layers=layers,
            num_neighbors=num_neighbors,
        )
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.embeddings: list[torch.Tensor] = []

    def configure_optimizers(self) -> None:
        return

    def training_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> None:
        del args, kwargs  # These variables are not used.

        self.model.feature_extractor.eval()
        output = self.model(batch["image"], batch["image_original"])

        self.embeddings.append(output["embeddings"])

    def fit(self) -> None:
        logger.info("Aggregating the embedding extracted from the training set.")
        embeddings = torch.vstack(self.embeddings)

        logger.info(
            "Applying core-set subsampling to get the embedding. Original shape: %s",
            str(tuple(embeddings.shape)),
        )
        sampler = KCenterGreedy(embedding=embeddings, sampling_ratio=self.coreset_sampling_ratio)
        coreset = sampler.sample_coreset()

        logger.info("Coreset shape: %s", str(tuple(coreset.shape)))
        self.model.memory_bank = coreset

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> STEP_OUTPUT:
        # These variables are not used.
        del args, kwargs

        # Get anomaly maps and predicted scores from the model.
        output = self.model(batch["image"], batch["image_original"])

        # Add anomaly maps and predicted scores to the batch.
        batch["anomaly_maps"] = output["anomaly_map"]
        batch["pred_scores"] = output["pred_score"]

        return batch

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        return {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        return LearningType.ONE_CLASS
