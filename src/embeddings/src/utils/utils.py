import importlib
import random
from functools import partial
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def get_model(hparams: DictConfig) -> Tuple[torch.nn.Module, int]:
    model = load_obj(hparams.model.backbone.class_name)
    model = model(weights=hparams.model.backbone.params.weights)

    if "ResNet" in model.__class__.__name__:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, hparams.model.backbone.params.embedding_num)
    elif "EfficientNet" in model.__class__.__name__:
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(
            in_features, hparams.model.backbone.params.embedding_num
        )
    elif "RegNet" in model.__class__.__name__:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, hparams.model.backbone.params.embedding_num)
    elif "MNASNet" in model.__class__.__name__:
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(
            in_features, hparams.model.backbone.params.embedding_num
        )
    elif "MobileNet" in model.__class__.__name__:
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(
            in_features, hparams.model.backbone.params.embedding_num
        )
    elif "ShuffleNet" in model.__class__.__name__:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, hparams.model.backbone.params.embedding_num)
    else:
        raise ValueError("Backbone not supported")

    return model


def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
