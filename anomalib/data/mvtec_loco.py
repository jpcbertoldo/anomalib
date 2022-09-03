"""MVTec LOCO AD Dataset (CC BY-NC-SA 4.0).

Description:
    This script contains PyTorch Dataset, Dataloader and PyTorch
        Lightning DataModule for the MVTec LOCO AD dataset.

    If the dataset is not on the file system, the script downloads and extracts the dataset.

License:
    MVTec LOCO AD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).

Reference:

    - Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger:
      Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection
      and Localization; in: International Journal of Computer Vision, 2022,
      DOI: 10.1007/s11263-022-01578-9.

    - https://www.mvtec.com/company/research/datasets/mvtec-loco
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from urllib.request import urlretrieve

import albumentations as A
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas.core.frame import DataFrame
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import VisionDataset

from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import DownloadProgressBar, hash_check, read_image, read_mask
from anomalib.data.utils.download import tar_extract_all
from anomalib.pre_processing import PreProcessor

logger = logging.getLogger(__name__)


TASK_CLASSIFICATION = "classification"
TASK_SEGMENTATION = "segmentation"
TASKS = (TASK_CLASSIFICATION, TASK_SEGMENTATION)

SPLIT_TRAIN = "train"
# "validation" instead of "val" is an explicit choice because this will match the name of the folder
SPLIT_VALIDATION = "validation"
SPLIT_TEST = "test"
SPLITS = (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)

# each image is read upon demand
IMREAD_STRATEGY_ONTHEFLY = "onthefly"
# all images are pre-loaded into memory at initialization
IMREAD_STRATEGY_PRELOAD = "preload"
IMREAD_STRATEGIES = (IMREAD_STRATEGY_ONTHEFLY, IMREAD_STRATEGY_PRELOAD)

CATEGORY_BREAKFAST_BOX = "breakfast_box"
CATEGORY_JUICE_BOTTLE = "juice_bottle"
CATEGORY_PUSHPINS = "pushpins"
CATEGORY_SCREW_BAG = "screw_bag"
CATEGORY_SPLICING_CONNECTORS = "splicing_connectors"
CATEGORIES: Tuple[str, ...] = (
    CATEGORY_BREAKFAST_BOX,
    CATEGORY_JUICE_BOTTLE,
    CATEGORY_PUSHPINS,
    CATEGORY_SCREW_BAG,
    CATEGORY_SPLICING_CONNECTORS,
)

LABEL_NORMAL = 0
LABEL_ANOMALOUS = 1

SUPER_ANOTYPE_GOOD = "good"
SUPER_ANOTYPE_LOGICAL = "logical_anomalies"
SUPER_ANOTYPE_STRUCTURAL = "structural_anomalies"
SUPER_ANOTYPES: Tuple[str, ...] = (SUPER_ANOTYPE_GOOD, SUPER_ANOTYPE_LOGICAL, SUPER_ANOTYPE_STRUCTURAL)

ANOTYPE_GOOD = "good"

ANOTYPES_BREAKFAST_BOX: Tuple[str, ...] = (
    ANOTYPE_BB_GOOD := "good",
    ANOTYPE_BB_COMPARTMENTS_SWAPPED := "compartments_swapped",
    ANOTYPE_BB_MISSING_ALMONDS := "missing_almonds",
    ANOTYPE_BB_MISSING_BANANAS := "missing_bananas",
    ANOTYPE_BB_MISSING_CEREALS := "missing_cereals",
    ANOTYPE_BB_MISSING_CEREALS_AND_TOPPINGS := "missing_cereals_and_toppings",
    ANOTYPE_BB_MISSING_TOPPINGS := "missing_toppings",
    ANOTYPE_BB_NECTARINE_1_TANGERINE_1 := "nectarine_1_tangerine_1",
    ANOTYPE_BB_NECTARINES_0_TANGERINE_1 := "nectarines_0_tangerine_1",
    ANOTYPE_BB_NECTARINES_0_TANGERINES_0 := "nectarines_0_tangerines_0",
    ANOTYPE_BB_NECTARINES_0_TANGERINES_2 := "nectarines_0_tangerines_2",
    ANOTYPE_BB_NECTARINES_0_TANGERINES_3 := "nectarines_0_tangerines_3",
    ANOTYPE_BB_NECTARINES_0_TANGERINES_4 := "nectarines_0_tangerines_4",
    ANOTYPE_BB_NECTARINES_2_TANGERINE_1 := "nectarines_2_tangerine_1",
    ANOTYPE_BB_NECTARINES_3_TANGERINES_0 := "nectarines_3_tangerines_0",
    ANOTYPE_BB_OVERFLOW := "overflow",
    ANOTYPE_BB_UNDERFLOW := "underflow",
    ANOTYPE_BB_WRONG_RATIO := "wrong_ratio",
    ANOTYPE_BB_BOX_DAMAGED := "box_damaged",
    ANOTYPE_BB_CONTAMINATION := "contamination",
    ANOTYPE_BB_FRUIT_DAMAGED := "fruit_damaged",
    ANOTYPE_BB_MIXED_CEREALS := "mixed_cereals",
    ANOTYPE_BB_TOPPINGS_CRUSHED := "toppings_crushed",
)

ANOTYPES_JUICE_BOTTLE: Tuple[str, ...] = (
    ANOTYPE_JB_GOOD := "good",
    ANOTYPE_JB_EMPTY_BOTTLE := "empty_bottle",
    ANOTYPE_JB_MISPLACED_FRUIT_ICON := "misplaced_fruit_icon",
    ANOTYPE_JB_MISPLACED_LABEL_BOTTOM := "misplaced_label_bottom",
    ANOTYPE_JB_MISPLACED_LABEL_TOP := "misplaced_label_top",
    ANOTYPE_JB_MISSING_BOTTOM_LABEL := "missing_bottom_label",
    ANOTYPE_JB_MISSING_FRUIT_ICON := "missing_fruit_icon",
    ANOTYPE_JB_MISSING_TOP_LABEL := "missing_top_label",
    ANOTYPE_JB_SWAPPED_LABELS := "swapped_labels",
    ANOTYPE_JB_WRONG_FILL_LEVEL_NOT_ENOUGH := "wrong_fill_level_not_enough",
    ANOTYPE_JB_WRONG_FILL_LEVEL_TOO_MUCH := "wrong_fill_level_too_much",
    ANOTYPE_JB_WRONG_JUICE_TYPE := "wrong_juice_type",
    ANOTYPE_JB_CONTAMINATION := "contamination",
    ANOTYPE_JB_DAMAGED_LABEL := "damaged_label",
    ANOTYPE_JB_INCOMPLETE_FRUIT_ICON := "incomplete_fruit_icon",
    ANOTYPE_JB_JUICE_COLOR := "juice_color",
    ANOTYPE_JB_LABEL_TEXT_INCOMPLETE := "label_text_incomplete",
    ANOTYPE_JB_ROTATED_LABEL := "rotated_label",
    ANOTYPE_JB_UNKNOWN_FRUIT_ICON := "unknown_fruit_icon",
)

ANOTYPES_PUSHPINS: Tuple[str, ...] = (
    ANOTYPE_P_GOOD := "good",
    ANOTYPE_P_ADDITIONAL_1_PUSHPIN := "additional_1_pushpin",
    ANOTYPE_P_ADDITIONAL_2_PUSHPINS := "additional_2_pushpins",
    ANOTYPE_P_MISSING_PUSHPIN := "missing_pushpin",
    ANOTYPE_P_MISSING_SEPARATOR := "missing_separator",
    ANOTYPE_P_BROKEN := "broken",
    ANOTYPE_P_COLOR := "color",
    ANOTYPE_P_CONTAMINATION := "contamination",
    ANOTYPE_P_FRONT_BENT := "front_bent",
)

ANOTYPES_SCREW_BAG: Tuple[str, ...] = (
    ANOTYPE_SB_GOOD := "good",
    ANOTYPE_SB_ADDITIONAL_1_LONG_SCREW := "additional_1_long_screw",
    ANOTYPE_SB_ADDITIONAL_1_NUT_ := "additional_1_nut_",
    ANOTYPE_SB_ADDITIONAL_1_SHORT_SCREW := "additional_1_short_screw",
    ANOTYPE_SB_ADDITIONAL_1_WASHER_ := "additional_1_washer_",
    ANOTYPE_SB_ADDITIONAL_2_NUTS_ := "additional_2_nuts_",
    ANOTYPE_SB_ADDITIONAL_2_WASHERS_ := "additional_2_washers_",
    ANOTYPE_SB_MISSING_1_LONG_SCREW := "missing_1_long_screw",
    ANOTYPE_SB_MISSING_1_NUT := "missing_1_nut",
    ANOTYPE_SB_MISSING_1_SHORT_SCREW := "missing_1_short_screw",
    ANOTYPE_SB_MISSING_1_WASHER := "missing_1_washer",
    ANOTYPE_SB_MISSING_2_NUTS := "missing_2_nuts",
    ANOTYPE_SB_MISSING_2_WASHERS := "missing_2_washers",
    ANOTYPE_SB_SCREW_TOO_LONG := "screw_too_long",
    ANOTYPE_SB_SCREW_TOO_SHOR := "screw_too_shor",
    ANOTYPE_SB_SCREWS_1_VERY_SHORT := "screws_1_very_short",
    ANOTYPE_SB_SCREWS_2_VERY_SHORT := "screws_2_very_short",
    ANOTYPE_SB_BAG_BROKEN := "bag_broken",
    ANOTYPE_SB_COLOR := "color",
    ANOTYPE_SB_CONTAMINATION := "contamination",
    ANOTYPE_SB_PART_BROKEN := "part_broken",
)

ANOTYPES_SPLICING_CONNECTORS: Tuple[str, ...] = (
    ANOTYPE_SC_GOOD := "good",
    ANOTYPE_SC_CABLE_COLOR := "cable_color",
    ANOTYPE_SC_CABLE_CUT := "cable_cut",
    ANOTYPE_SC_CABLE_TOO_SHORT_T2 := "cable_too_short_t2",
    ANOTYPE_SC_CABLE_TOO_SHORT_T3 := "cable_too_short_t3",
    ANOTYPE_SC_CABLE_TOO_SHORT_T5 := "cable_too_short_t5",
    ANOTYPE_SC_EXTRA_CABLE := "extra_cable",
    ANOTYPE_SC_MISSING_CABLE := "missing_cable",
    ANOTYPE_SC_MISSING_CONNECTOR := "missing_connector",
    ANOTYPE_SC_MISSING_CONNECTOR_AND_CABLE := "missing_connector_and_cable",
    ANOTYPE_SC_WRONG_CABLE_LOCATION := "wrong_cable_location",
    ANOTYPE_SC_WRONG_CONNECTOR_TYPE_3_2 := "wrong_connector_type_3_2",
    ANOTYPE_SC_WRONG_CONNECTOR_TYPE_5_2 := "wrong_connector_type_5_2",
    ANOTYPE_SC_WRONG_CONNECTOR_TYPE_5_3 := "wrong_connector_type_5_3",
    ANOTYPE_SC_BROKEN_CABLE := "broken_cable",
    ANOTYPE_SC_BROKEN_CONNECTOR := "broken_connector",
    ANOTYPE_SC_CABLE_NOT_PLUGGED := "cable_not_plugged",
    ANOTYPE_SC_COLOR := "color",
    ANOTYPE_SC_CONTAMINATION := "contamination",
    ANOTYPE_SC_FLIPPED_CONNECTOR := "flipped_connector",
    ANOTYPE_SC_OPEN_LEVER := "open_lever",
    ANOTYPE_SC_UNKNOWN_CABLE_COLOR := "unknown_cable_color",
)

# this is given at the paper, each anomaly type (label) has a different gtvalue in the mask
# source: Beyond Dents and Scratches: Logical Constraints in
#   Unsupervised Anomaly Detection and Localization (Bergmann, P. et al, 2022).
_MAP_ANOTYPE_2_GTVALUE: Dict[Tuple[str, str, str], int] = {
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_LOGICAL, ANOTYPE_BB_COMPARTMENTS_SWAPPED): 242,
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_LOGICAL, ANOTYPE_BB_MISSING_ALMONDS): 255,
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_LOGICAL, ANOTYPE_BB_MISSING_BANANAS): 254,
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_LOGICAL, ANOTYPE_BB_MISSING_CEREALS): 252,
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_LOGICAL, ANOTYPE_BB_MISSING_CEREALS_AND_TOPPINGS): 251,
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_LOGICAL, ANOTYPE_BB_MISSING_TOPPINGS): 253,
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_LOGICAL, ANOTYPE_BB_NECTARINE_1_TANGERINE_1): 249,
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_LOGICAL, ANOTYPE_BB_NECTARINES_0_TANGERINE_1): 245,
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_LOGICAL, ANOTYPE_BB_NECTARINES_0_TANGERINES_0): 244,
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_LOGICAL, ANOTYPE_BB_NECTARINES_0_TANGERINES_2): 248,
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_LOGICAL, ANOTYPE_BB_NECTARINES_0_TANGERINES_3): 247,
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_LOGICAL, ANOTYPE_BB_NECTARINES_0_TANGERINES_4): 243,
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_LOGICAL, ANOTYPE_BB_NECTARINES_2_TANGERINE_1): 250,
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_LOGICAL, ANOTYPE_BB_NECTARINES_3_TANGERINES_0): 246,
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_LOGICAL, ANOTYPE_BB_OVERFLOW): 241,
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_LOGICAL, ANOTYPE_BB_UNDERFLOW): 240,
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_LOGICAL, ANOTYPE_BB_WRONG_RATIO): 239,
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_BB_BOX_DAMAGED): 236,
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_BB_CONTAMINATION): 234,
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_BB_FRUIT_DAMAGED): 237,
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_BB_MIXED_CEREALS): 238,
    (CATEGORY_BREAKFAST_BOX, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_BB_TOPPINGS_CRUSHED): 235,
    (CATEGORY_JUICE_BOTTLE, SUPER_ANOTYPE_LOGICAL, ANOTYPE_JB_EMPTY_BOTTLE): 247,
    (CATEGORY_JUICE_BOTTLE, SUPER_ANOTYPE_LOGICAL, ANOTYPE_JB_MISPLACED_FRUIT_ICON): 244,
    (CATEGORY_JUICE_BOTTLE, SUPER_ANOTYPE_LOGICAL, ANOTYPE_JB_MISPLACED_LABEL_BOTTOM): 249,
    (CATEGORY_JUICE_BOTTLE, SUPER_ANOTYPE_LOGICAL, ANOTYPE_JB_MISPLACED_LABEL_TOP): 250,
    (CATEGORY_JUICE_BOTTLE, SUPER_ANOTYPE_LOGICAL, ANOTYPE_JB_MISSING_BOTTOM_LABEL): 254,
    (CATEGORY_JUICE_BOTTLE, SUPER_ANOTYPE_LOGICAL, ANOTYPE_JB_MISSING_FRUIT_ICON): 243,
    (CATEGORY_JUICE_BOTTLE, SUPER_ANOTYPE_LOGICAL, ANOTYPE_JB_MISSING_TOP_LABEL): 255,
    (CATEGORY_JUICE_BOTTLE, SUPER_ANOTYPE_LOGICAL, ANOTYPE_JB_SWAPPED_LABELS): 253,
    (CATEGORY_JUICE_BOTTLE, SUPER_ANOTYPE_LOGICAL, ANOTYPE_JB_WRONG_FILL_LEVEL_NOT_ENOUGH): 245,
    (CATEGORY_JUICE_BOTTLE, SUPER_ANOTYPE_LOGICAL, ANOTYPE_JB_WRONG_FILL_LEVEL_TOO_MUCH): 246,
    (CATEGORY_JUICE_BOTTLE, SUPER_ANOTYPE_LOGICAL, ANOTYPE_JB_WRONG_JUICE_TYPE): 240,
    (CATEGORY_JUICE_BOTTLE, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_JB_CONTAMINATION): 238,
    (CATEGORY_JUICE_BOTTLE, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_JB_DAMAGED_LABEL): 252,
    (CATEGORY_JUICE_BOTTLE, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_JB_INCOMPLETE_FRUIT_ICON): 241,
    (CATEGORY_JUICE_BOTTLE, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_JB_JUICE_COLOR): 239,
    (CATEGORY_JUICE_BOTTLE, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_JB_LABEL_TEXT_INCOMPLETE): 248,
    (CATEGORY_JUICE_BOTTLE, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_JB_ROTATED_LABEL): 251,
    (CATEGORY_JUICE_BOTTLE, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_JB_UNKNOWN_FRUIT_ICON): 242,
    (CATEGORY_PUSHPINS, SUPER_ANOTYPE_LOGICAL, ANOTYPE_P_ADDITIONAL_1_PUSHPIN): 255,
    (CATEGORY_PUSHPINS, SUPER_ANOTYPE_LOGICAL, ANOTYPE_P_ADDITIONAL_2_PUSHPINS): 254,
    (CATEGORY_PUSHPINS, SUPER_ANOTYPE_LOGICAL, ANOTYPE_P_MISSING_PUSHPIN): 253,
    (CATEGORY_PUSHPINS, SUPER_ANOTYPE_LOGICAL, ANOTYPE_P_MISSING_SEPARATOR): 252,
    (CATEGORY_PUSHPINS, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_P_BROKEN): 250,
    (CATEGORY_PUSHPINS, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_P_COLOR): 249,
    (CATEGORY_PUSHPINS, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_P_CONTAMINATION): 248,
    (CATEGORY_PUSHPINS, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_P_FRONT_BENT): 251,
    (CATEGORY_SCREW_BAG, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SB_ADDITIONAL_1_LONG_SCREW): 251,
    (CATEGORY_SCREW_BAG, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SB_ADDITIONAL_1_NUT_): 249,
    (CATEGORY_SCREW_BAG, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SB_ADDITIONAL_1_SHORT_SCREW): 250,
    (CATEGORY_SCREW_BAG, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SB_ADDITIONAL_1_WASHER_): 247,
    (CATEGORY_SCREW_BAG, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SB_ADDITIONAL_2_NUTS_): 248,
    (CATEGORY_SCREW_BAG, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SB_ADDITIONAL_2_WASHERS_): 246,
    (CATEGORY_SCREW_BAG, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SB_MISSING_1_LONG_SCREW): 245,
    (CATEGORY_SCREW_BAG, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SB_MISSING_1_NUT): 243,
    (CATEGORY_SCREW_BAG, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SB_MISSING_1_SHORT_SCREW): 244,
    (CATEGORY_SCREW_BAG, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SB_MISSING_1_WASHER): 241,
    (CATEGORY_SCREW_BAG, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SB_MISSING_2_NUTS): 242,
    (CATEGORY_SCREW_BAG, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SB_MISSING_2_WASHERS): 240,
    (CATEGORY_SCREW_BAG, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SB_SCREW_TOO_LONG): 255,
    (CATEGORY_SCREW_BAG, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SB_SCREW_TOO_SHOR): 254,
    (CATEGORY_SCREW_BAG, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SB_SCREWS_1_VERY_SHORT): 253,
    (CATEGORY_SCREW_BAG, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SB_SCREWS_2_VERY_SHORT): 252,
    (CATEGORY_SCREW_BAG, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_SB_BAG_BROKEN): 239,
    (CATEGORY_SCREW_BAG, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_SB_COLOR): 238,
    (CATEGORY_SCREW_BAG, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_SB_CONTAMINATION): 237,
    (CATEGORY_SCREW_BAG, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_SB_PART_BROKEN): 236,
    (CATEGORY_SPLICING_CONNECTORS, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SC_CABLE_COLOR): 245,
    (CATEGORY_SPLICING_CONNECTORS, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SC_CABLE_CUT): 243,
    (CATEGORY_SPLICING_CONNECTORS, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SC_CABLE_TOO_SHORT_T2): 252,
    (CATEGORY_SPLICING_CONNECTORS, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SC_CABLE_TOO_SHORT_T3): 251,
    (CATEGORY_SPLICING_CONNECTORS, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SC_CABLE_TOO_SHORT_T5): 250,
    (CATEGORY_SPLICING_CONNECTORS, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SC_EXTRA_CABLE): 246,
    (CATEGORY_SPLICING_CONNECTORS, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SC_MISSING_CABLE): 247,
    (CATEGORY_SPLICING_CONNECTORS, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SC_MISSING_CONNECTOR): 249,
    (CATEGORY_SPLICING_CONNECTORS, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SC_MISSING_CONNECTOR_AND_CABLE): 248,
    (CATEGORY_SPLICING_CONNECTORS, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SC_WRONG_CABLE_LOCATION): 240,
    (CATEGORY_SPLICING_CONNECTORS, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SC_WRONG_CONNECTOR_TYPE_3_2): 253,
    (CATEGORY_SPLICING_CONNECTORS, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SC_WRONG_CONNECTOR_TYPE_5_2): 255,
    (CATEGORY_SPLICING_CONNECTORS, SUPER_ANOTYPE_LOGICAL, ANOTYPE_SC_WRONG_CONNECTOR_TYPE_5_3): 254,
    (CATEGORY_SPLICING_CONNECTORS, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_SC_BROKEN_CABLE): 244,
    (CATEGORY_SPLICING_CONNECTORS, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_SC_BROKEN_CONNECTOR): 238,
    (CATEGORY_SPLICING_CONNECTORS, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_SC_CABLE_NOT_PLUGGED): 242,
    (CATEGORY_SPLICING_CONNECTORS, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_SC_COLOR): 236,
    (CATEGORY_SPLICING_CONNECTORS, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_SC_CONTAMINATION): 235,
    (CATEGORY_SPLICING_CONNECTORS, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_SC_FLIPPED_CONNECTOR): 239,
    (CATEGORY_SPLICING_CONNECTORS, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_SC_OPEN_LEVER): 237,
    (CATEGORY_SPLICING_CONNECTORS, SUPER_ANOTYPE_STRUCTURAL, ANOTYPE_SC_UNKNOWN_CABLE_COLOR): 241,
}

# map: (category, gtvalue) -> (superlabel, label)
_MAP_GTVALUE_2_ANOTYPE: Dict[Tuple[str, int], Tuple[str, str]] = {
    (category, gtvalue): (super_anotype, anotype)
    for (category, super_anotype, anotype), gtvalue in _MAP_ANOTYPE_2_GTVALUE.items()
}

# expected number of images in each category split so that we can check if the dataset is complete
# source: Beyond Dents and Scratches: Logical Constraints
#   in Unsupervised Anomaly Detection and Localization (Bergmann, P. et al, 2022).
_EXPECTED_NSAMPLES: Dict[Tuple[str, str], int] = {
    (CATEGORY_BREAKFAST_BOX, SPLIT_TRAIN): 351,
    (CATEGORY_BREAKFAST_BOX, SPLIT_VALIDATION): 62,
    (CATEGORY_BREAKFAST_BOX, SPLIT_TEST): 275,
    (CATEGORY_JUICE_BOTTLE, SPLIT_TRAIN): 335,
    (CATEGORY_JUICE_BOTTLE, SPLIT_VALIDATION): 54,
    (CATEGORY_JUICE_BOTTLE, SPLIT_TEST): 330,
    (CATEGORY_PUSHPINS, SPLIT_TRAIN): 372,
    (CATEGORY_PUSHPINS, SPLIT_VALIDATION): 69,
    (CATEGORY_PUSHPINS, SPLIT_TEST): 310,
    (CATEGORY_SCREW_BAG, SPLIT_TRAIN): 360,
    (CATEGORY_SCREW_BAG, SPLIT_VALIDATION): 60,
    (CATEGORY_SCREW_BAG, SPLIT_TEST): 341,
    # these two below were wrong in the paper
    # (CATEGORY_SPLICING_CONNECTORS, SPLIT_TRAIN): 354,
    # (CATEGORY_SPLICING_CONNECTORS, SPLIT_VALIDATION): 59,
    (CATEGORY_SPLICING_CONNECTORS, SPLIT_TRAIN): 360,
    (CATEGORY_SPLICING_CONNECTORS, SPLIT_VALIDATION): 60,
    (CATEGORY_SPLICING_CONNECTORS, SPLIT_TEST): 312,
}


def _binarize_mask_float(mask: np.ndarray) -> np.ndarray:  # noqa
    """
    the masks use different gtvalue values for the different anomaly types so the > 0 is making it binary
    this operation is very simple but it is in a function to make
    sure its standard because it is used in different places
    e.g. preloading while building the dataset and on the fly while training
    """
    return (mask > 0).astype(float)


def _make_dataset(
    path: Path,
    split: Optional[str] = None,
    imread_strategy: str = IMREAD_STRATEGY_PRELOAD,
) -> DataFrame:  # noqa D212
    """
    Find the images in the given path and create a DataFrame with all the information from each sample.

    Expected structure of the files in the dataset ("/" is 'path')

    images: /{split}/{super_anotype}/{image_index}.png

    /train/good/000.png
    /train/good/...
    /train/good/350.png

    /validation/good/...

    /test/good/...
    /test/logical_anomalies/...
    /test/structural_anomalies/...

    masks: /ground_truth/{super_anotype}/{image_index}/000.png

    /ground_truth/logical_anomalies/000/000.png
    /ground_truth/logical_anomalies/.../000.png
    /ground_truth/logical_anomalies/079/000.png

    /ground_truth/structural_anomalies/.../000.png
    ...
    """
    # todo create optional to get a subset of anomlies in the test

    assert split is None or split in SPLITS, f"Invalid split: {split}"
    assert imread_strategy in IMREAD_STRATEGIES, f"Invalid imread strategy: {imread_strategy}"

    category = path.resolve().name
    assert category in CATEGORIES, f"Invalid path '{path}'. The directory ('{category}') must be one of {CATEGORIES}"

    if split is None:
        return pd.concat([_make_dataset(path, split_, imread_strategy) for split_ in SPLITS], axis=0)

    logger.info("Creating MVTec LOCO AD dataset for category '%s' split '%s'", category, split)

    # these values look like "(train|validation|test)/(good|logical_anomalies|structural_anomalies)/(000|...|n).png"
    # where (a|b) means either a or b
    samples_paths = sorted(path.glob(f"{split}/**/*.png"))
    expected_nsamples = _EXPECTED_NSAMPLES[(category, split)]

    if len(samples_paths) != expected_nsamples:
        warnings.warn(
            f"Expected {expected_nsamples} samples for split '{split}' "
            "in category '{category}' but found {len(samples_paths)}."
            "Is the dataset corrupted?"
        )

    def build_record(sample_path: Path):

        ret: Dict[str, Union[Path, None, str, int]] = {
            "image_path": sample_path,
            **dict(zip(("split", "super_anotype", "image_filename"), sample_path.parts[-3:])),
        }

        super_anotype: str = ret["super_anotype"]  # type: ignore

        if super_anotype == SUPER_ANOTYPE_GOOD:
            ret.update(
                {
                    "mask_path": None,
                    "label": LABEL_NORMAL,
                    "super_anotype": SUPER_ANOTYPE_GOOD,
                    "anotype": ANOTYPE_GOOD,
                }
            )
            return ret

        if super_anotype in (SUPER_ANOTYPE_LOGICAL, SUPER_ANOTYPE_STRUCTURAL):

            mask_path: Path = path / "ground_truth" / super_anotype / sample_path.stem / "000.png"

            assert mask_path.exists(), f"Mask file '{mask_path}' does not exist. Is the dataset corrupted?"

            # mask images are supposed to have only two values: 0 and GTVALUE_ANOMALY
            # GTVALUE_ANOMALY \in {234, ..., 255} and is given in the paper, encoded in the mapping below
            # TODO create an issue to cache this info so the mask is not read here
            gtvalue = read_mask(mask_path).astype(int).max()
            _, anotype = _MAP_GTVALUE_2_ANOTYPE[(category, gtvalue)]

            ret.update(
                {
                    "mask_path": mask_path,
                    "gtvalue": gtvalue,
                    "label": LABEL_ANOMALOUS,
                    "super_anotype": super_anotype,
                    "anotype": anotype,
                }
            )

            return ret

        # there should only be the folders "good", "logical_anomalies" and "structural_anomalies"
        raise RuntimeError(f"Something wrong in the dataset folder. Unknown folder {super_anotype}, path={sample_path}")

    samples = pd.DataFrame.from_records([build_record(sp) for sp in samples_paths])

    if imread_strategy == IMREAD_STRATEGY_PRELOAD:

        warnings.warn(
            "Preloading images into memory. "
            "If your dataset is too large, consider using another imread_strategy instead.",
            stacklevel=3,
        )

        logger.debug("Preloading images into memory")
        samples["image"] = samples.image_path.map(read_image)

        logger.debug("Preloading masks into memory")

        # this is used to select the rows in the dataframe
        has_mask = ~samples.mask_path.isnull()

        samples.loc[has_mask, "mask"] = samples.loc[has_mask, "mask_path"].map(
            lambda x: _binarize_mask_float(read_mask(x))
        )
        samples.loc[~has_mask, "mask"] = None

    return samples


class MVTecLOCODataset(VisionDataset):
    """MVTec LOCO AD PyTorch Dataset."""

    def __init__(
        self,
        root: Union[Path, str],
        category: str,
        pre_process: PreProcessor,
        split: str,
        task: str = TASK_SEGMENTATION,
        # create_validation_set: bool = False,
        imread_strategy: str = IMREAD_STRATEGY_PRELOAD,
    ) -> None:
        super().__init__(root)

        assert split in SPLITS, f"Split '{split}' is not supported. Supported splits are {SPLITS}"
        assert task in TASKS, f"Task '{task}' is not supported. Supported tasks are {TASKS}"
        assert (
            imread_strategy in IMREAD_STRATEGIES
        ), f"Imread strategy '{imread_strategy}' is not supported. Supported imread strategies are {IMREAD_STRATEGIES}"

        self.root = Path(root) if isinstance(root, str) else root
        self.category: str = category
        self.split = split
        self.task = task
        self.pre_process = pre_process
        self.imread_strategy = imread_strategy

        self.samples = _make_dataset(
            path=self.category_dataset_path,
            split=self.split,
            imread_strategy=self.imread_strategy,
        )

    @property
    def category_dataset_path(self) -> Path:
        """Path to the category dataset (root/category) folder."""
        return self.root / self.category

    def __len__(self) -> int:
        """Get length of the dataset."""
        return len(self.samples)

    def _get_image(self, index: int) -> ndarray:
        """Get image at index."""

        if self.imread_strategy == IMREAD_STRATEGY_PRELOAD:
            return self.samples.image[index]

        if self.imread_strategy == IMREAD_STRATEGY_ONTHEFLY:
            return read_image(self.samples.image_path[index])

        raise NotImplementedError(f"Imread strategy '{self.imread_strategy}' is not supported.")

    def _get_mask(self, index: int) -> ndarray:
        """Get mask at index."""

        if self.imread_strategy == IMREAD_STRATEGY_PRELOAD:
            return self.samples.mask[index]

        if self.imread_strategy == IMREAD_STRATEGY_ONTHEFLY:
            return _binarize_mask_float(read_mask(self.samples.mask_path[index]))

        raise NotImplementedError(f"Imread strategy '{self.imread_strategy}' is not supported.")

    def __getitem__(self, index: int) -> Dict[str, Union[str, Tensor]]:
        """Get dataset item for the index ``index``.

        Args:
            index (int): Index to get the item.

        Returns:
            Union[Dict[str, Tensor], Dict[str, Union[str, Tensor]]]: Dict of image tensor during training.
                Otherwise, Dict containing image path, target path, image tensor, label and transformed bounding box.
        """
        item: Dict[str, Union[str, Tensor]] = {}

        image = self._get_image(index)
        pre_processed = self.pre_process(image=image)
        item = {
            "image": pre_processed["image"],
        }

        if self.split not in (SPLIT_VALIDATION, SPLIT_TEST):
            return item

        item.update(
            {
                "label": self.samples.label[index],
                "image_path": self.samples.image_path[index],
                "anotype": self.samples.anotype[index],
                "super_anotype": self.samples.super_anotype[index],
            }
        )

        if self.task != TASK_SEGMENTATION:
            return item

        # Only Anomalous (1) images has masks in MVTec LOCO AD dataset.
        # Therefore, create empty mask for Normal (0) images.
        if self.samples.label[index] == LABEL_NORMAL:
            mask = np.zeros(shape=image.shape[:2])  # shape: (H, W, C)

        else:
            mask = self._get_mask(index)

        pre_processed = self.pre_process(image=image, mask=mask)
        item.update(
            {
                "mask_path": self.samples.mask_path[index],
                "mask": pre_processed["mask"],
            }
        )

        return item


@DATAMODULE_REGISTRY
class MVTecLOCO(LightningDataModule):
    """MVTec LOCO AD Lightning Data Module."""

    # todo correct inconsistency: `transform_config_*val*` used for val and
    #   test set, but `*test*_batch_size` used for val and set

    def __init__(
        self,
        root: str,
        category: str,
        # TODO: add a parameter to specify the anomaly types and (more specifically)
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        train_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 8,
        task: str = TASK_SEGMENTATION,
        transform_config_train: Optional[Union[str, A.Compose]] = None,
        transform_config_val: Optional[Union[str, A.Compose]] = None,
        seed: Optional[int] = None,
        imread_strategy: str = IMREAD_STRATEGY_PRELOAD,
    ) -> None:
        super().__init__()

        # todo? add input assertions/validations
        assert task in TASKS, f"Task '{task}' is not supported. Supported tasks are {TASKS}"
        assert (
            imread_strategy in IMREAD_STRATEGIES
        ), f"Imread strategy '{imread_strategy}' is not supported. Supported imread strategies are {IMREAD_STRATEGIES}"

        self.root = root if isinstance(root, Path) else Path(root)
        self.category = category
        self.transform_config_train = transform_config_train
        self.transform_config_val = transform_config_val
        self.image_size = image_size

        self.pre_process_train = PreProcessor(config=self.transform_config_train, image_size=self.image_size)
        self.pre_process_val = PreProcessor(config=self.transform_config_val, image_size=self.image_size)

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.task = task
        self.seed = seed
        self.imread_strategy = imread_strategy

        self.train_data: Dataset
        self.test_data: Dataset
        self.val_data: Dataset
        self.inference_data: Dataset

    @property
    def category_dataset_path(self) -> Path:
        """Path to the category dataset (root/category) folder."""
        return self.root / self.category

    def prepare_data(self) -> None:
        """Download the dataset if not available."""

        if self.category_dataset_path.is_dir():
            logger.info("Found the dataset.")

        else:
            self.root.mkdir(parents=True, exist_ok=True)

            logger.info("Downloading the Mvtec LOCO AD dataset.")
            # flake8: noqa: E501
            # pylint: disable=line-too-long
            url_mvtec_loco_targz = "https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701/mvtec_loco_anomaly_detection.tar.xz"
            dataset_name = "mvtec_loco_anomaly_detection.tar.xz"
            zip_filename = self.root / dataset_name
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="MVTec LOCO download") as progress_bar:
                urlretrieve(
                    url=url_mvtec_loco_targz,
                    filename=zip_filename,
                    reporthook=progress_bar.update_to,
                )

            logger.info("Checking hash")
            md5hash_mvtec_loco = "d40f092ac6f88433f609583c4a05f56f"
            hash_check(zip_filename, md5hash_mvtec_loco)

            logger.info("Extracting the dataset.")
            logger.debug("Extracting to %s", self.root)
            tar_extract_all(zip_filename, self.root)

            logger.info("Cleaning the tar file")
            zip_filename.unlink()

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup train, validation and test data.

        Args:
          stage: Optional[str]:  fit/validate/test/predict stages. (Default value = None = fit)

        """
        # pylint: disable=consider-using-f-string
        logger.info("Setting up %s dataset." % stage or TrainerFn.FITTING)

        if stage in (None, TrainerFn.FITTING):
            self.train_data = MVTecLOCODataset(
                root=self.root,
                category=self.category,
                split=SPLIT_TRAIN,
                pre_process=self.pre_process_train,
                task=self.task,
                imread_strategy=self.imread_strategy,
            )

        if stage == TrainerFn.VALIDATING:
            self.val_data = MVTecLOCODataset(
                root=self.root,
                category=self.category,
                pre_process=self.pre_process_val,
                split=SPLIT_VALIDATION,
                task=self.task,
                imread_strategy=self.imread_strategy,
            )

        if stage == TrainerFn.TESTING:
            self.test_data = MVTecLOCODataset(
                root=self.root,
                category=self.category,
                pre_process=self.pre_process_val,
                split=SPLIT_TEST,
                task=self.task,
                imread_strategy=self.imread_strategy,
            )

        if stage == TrainerFn.PREDICTING:
            self.inference_data = InferenceDataset(
                path=self.root, image_size=self.image_size, transform_config=self.transform_config_val
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get train dataloader."""
        return DataLoader(self.train_data, shuffle=True, batch_size=self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Get validation dataloader."""
        return DataLoader(self.val_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get test dataloader."""
        return DataLoader(self.test_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """Get predict dataloader."""
        return DataLoader(
            self.inference_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers
        )


# TODO remove me
# debug _make_dataset in main
if __name__ == "__main__":
    import itertools

    for cat in CATEGORIES:
        _make_dataset(Path(f"/home/jcasagrandebertoldo/Downloads/loco/{cat}"))

    for cat, split_ in itertools.product(CATEGORIES, SPLITS):
        pass
        # _make_dataset(Path(f"/home/jcasagrandebertoldo/Downloads/loco/{cat}"), split_)
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # test instantiate of the two classes
        # then test with script
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next
        # next