"""Feature Extractor based on TorchFX."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
from torch import Tensor, nn
from torch.fx.graph_module import GraphModule
from torchvision.models._api import WeightsEnum
from torchvision.models.feature_extraction import create_feature_extractor


@dataclass
class BackboneParams:
    """Used for serializing the backbone."""
    
    class_path: Union[str, nn.Module]
    init_args: Dict = field(default_factory=dict, repr=False)
    
    # automatically set in __post_init__
    class_name: str = field(init=False)
    module_path: Union[str, Path] = field(init=False)
    
    @ property
    def fully_qualified_class_name(self) -> str:
        return f"{self.module_path}.{self.class_name}"
    
    def __post_init__(self):
        
        if isinstance(self.class_path, str):
            
            # if it has a "::" then it is a ".py" path with a class name
            if "::" in self.class_path:
                module_path, class_name = self.class_path.split("::")
                module_path = Path(module_path)
                assert (module_path.is_file() and module_path.suffix == ".py") or (module_path / "__init__.py").is_file(), f"{module_path} is not a valid path to a python module"
                self.module_path = module_path
                self.class_name = class_name
                return
                
            path_parts = self.class_path.split(".")
            
            if len(path_parts) == 1:
                self.module_path = "torchvision.models"
                self.class_name = path_parts[0]
            
            else:
                module_path = ".".join(path_parts[:-1])
                assert not module_path.endswith(".py"), "Module is should be in the python path, '.py' is unnecessary!"
                self.module_path = module_path
                self.class_name = path_parts[-1]
        
        elif isinstance(self.class_path, nn.Module):
            
                self.module_path = self.class_path.__module__
                self.class_name = self.class_path.__class__.__name__
            
        else:
            raise ValueError(f"Invalid class_path type: {type(self.class_path)}")
    
    
def _get_class(class_name: str, module_path: Union[str, Path]) -> Callable[..., nn.Module]:
    """Get the backbone class.
    """
    
    if isinstance(module_path, Path) or module_path.endswith(".py"):
        module_path = Path(module_path)
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_path)
    
    try:
        return getattr(module, class_name)
            
    except ModuleNotFoundError as exception:
        raise ModuleNotFoundError(
            f"Backbone {class_name} not found in torchvision.models nor in {module_path} module."
        ) from exception


class TorchFXFeatureExtractor(nn.Module):
    """Extract features from a CNN.

    Args:
        backbone (Union[str, BackboneParams, Dict, nn.Module]): The backbone to which the feature extraction hooks are
            attached. If the name is provided, the model is loaded from torchvision. Otherwise, the model class can be
            provided and it will try to load the weights from the provided weights file.
        return_nodes (Iterable[str]): List of layer names of the backbone to which the hooks are attached.
            You can find the names of these nodes by using ``get_graph_node_names`` function.
        weights (Optional[Union[WeightsEnum,str]]): Weights enum to use for the model. Torchvision models require
            ``WeightsEnum``. These enums are defined in ``torchvision.models.<model>``. You can pass the weights
            path for custom models.
        requires_grad (bool): Models like ``stfpm`` use the feature extractor for training. In such cases we should
            set ``requires_grad`` to ``True``. Default is ``False``.

    Example:
        With torchvision models:

            >>> import torch
            >>> from anomalib.models.components.feature_extractors import TorchFXFeatureExtractor
            >>> from torchvision.models.efficientnet import EfficientNet_B5_Weights
            >>> feature_extractor = TorchFXFeatureExtractor(
                    backbone="efficientnet_b5",
                    return_nodes=["features.6.8"],
                    weights=EfficientNet_B5_Weights.DEFAULT
                )
            >>> input = torch.rand((32, 3, 256, 256))
            >>> features = feature_extractor(input)
            >>> [layer for layer in features.keys()]
                ["features.6.8"]
            >>> [feature.shape for feature in features.values()]
                [torch.Size([32, 304, 8, 8])]

        With custom models:

            >>> from anomalib.models.components.feature_extractors import TorchFXFeatureExtractor
            >>> feature_extractor = TorchFXFeatureExtractor(
                    "path.to.CustomModel", ["linear_relu_stack.3"], weights="path/to/weights.pth"
                )
            >>> input = torch.randn(1, 1, 28, 28)
            >>> features = feature_extractor(input)
            >>> [layer for layer in features.keys()]
                ["linear_relu_stack.3"]
    """

    def __init__(
        self,
        backbone: Union[str, BackboneParams, Dict, nn.Module],
        return_nodes: List[str],
        weights: Optional[Union[WeightsEnum, str]] = None,
        requires_grad: bool = False,
    ):
        super().__init__()
        
        if isinstance(backbone, nn.Module):
            
            backbone_model = backbone
            
            if weights is not None:
                raise ValueError("Weights should be None when backbone is an instance of nn.Module")
        
        else: 
            
            if isinstance(backbone, dict):
                backbone = BackboneParams(**backbone)
                
            elif not isinstance(backbone, BackboneParams):  # if str or nn.Module
                backbone = BackboneParams(class_path=backbone)

            backbone_class = _get_class(backbone.class_name, backbone.module_path)
            
            # torchvision models use WeightsEnum and have the weights as a parameter of the init
            init_args = {**backbone.init_args, **(dict(weights=weights) if isinstance(weights, WeightsEnum) else {})}
            backbone_model = backbone_class(**init_args)
        
            if isinstance(weights, str):  # custom model weights in a file
                model_weights = torch.load(weights)
                if "state_dict" in model_weights:
                    model_weights = model_weights["state_dict"]
                backbone_model.load_state_dict(model_weights)
            
        feature_extractor = create_feature_extractor(model=backbone_model, return_nodes=return_nodes)

        if not requires_grad:
            feature_extractor.eval()
            for param in feature_extractor.parameters():
                param.requires_grad_(False)
            
        else:
            feature_extractor.train()
            for param in feature_extractor.parameters():
                param.requires_grad_(True)

        self.feature_extractor = feature_extractor

        #     self.feature_extractor = self.initialize_feature_extractor(backbone, return_nodes, weights, requires_grad)

        # def initialize_feature_extractor(
        #     self,
        #     backbone: BackboneParams,
        #     return_nodes: List[str],
        #     weights: Optional[Union[WeightsEnum, str]] = None,
        #     requires_grad: bool = False,
        # ) -> Union[GraphModule, nn.Module]:

        # """Extract features from a CNN.

        # Args:
        #     backbone (Union[str, BackboneParams]): The backbone to which the feature extraction hooks are attached.
        #         If the name is provided, the model is loaded from torchvision. Otherwise, the model class can be
        #         provided and it will try to load the weights from the provided weights file.
        #     return_nodes (Iterable[str]): List of layer names of the backbone to which the hooks are attached.
        #         You can find the names of these nodes by using ``get_graph_node_names`` function.
        #     weights (Optional[Union[WeightsEnum,str]]): Weights enum to use for the model. Torchvision models require
        #         ``WeightsEnum``. These enums are defined in ``torchvision.models.<model>``. You can pass the weights
        #         path for custom models.
        #     requires_grad (bool): Models like ``stfpm`` use the feature extractor for training. In such cases we should
        #         set ``requires_grad`` to ``True``. Default is ``False``.

        # Returns:
        #     Feature Extractor based on TorchFX.
        # """
        
    def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
        """Extract features from the input."""
        return self.feature_extractor(inputs)
