import importlib
import os
import re
from collections import OrderedDict
from functools import partial
from typing import Any, Tuple, Union

import torch
from torchvision import models


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


class EmbeddingInferencer(torch.nn.Module):
    architectures = re.compile(
        "|".join(
            ["resnet50", "efficientnet", "mnasnet", "regnet", "shufflenet", "mobilenet"]
        )
    )

    def __init__(self, model_ckpt: Union[str, os.PathLike], embedding_size: int):
        super(EmbeddingInferencer, self).__init__()
        architecture = self.architectures.findall(model_ckpt)[-1]
        state_dict = torch.load(model_ckpt)
        # state_dict = OrderedDict({k[6:]: v for k, v in state_dict.items()})
        fc_layer_template = partial(torch.nn.Linear, out_features=embedding_size)
        if architecture == "resnet50":
            self.model = models.resnet50(weights=None)
            in_features = self.model.fc.in_features
            self.model.fc = fc_layer_template(in_features=in_features)
        elif architecture == "efficientnet":
            self.model = models.efficientnet_b0(weights=None)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = fc_layer_template(in_features=in_features)
        elif architecture == "mnasnet":
            self.model = models.mnasnet0_5(weights=None)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = fc_layer_template(in_features=in_features)
        elif architecture == "regnet":
            self.model = models.regnet_y_400mf(weights=None)
            in_features = self.model.fc.in_features
            self.model.fc = fc_layer_template(in_features=in_features)
        elif architecture == "shufflenet":
            self.model = models.shufflenet_v2_x0_5(weights=None)
            in_features = self.model.fc.in_features
            self.model.fc = fc_layer_template(in_features=in_features)
        elif architecture == "mobilenet":
            self.model = models.mobilenet_v2(weights=None)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = fc_layer_template(in_features=in_features)
        else:
            raise ValueError("Architecture is not supported!")
        self.model.load_state_dict(state_dict=state_dict)
        self.model.eval()

    def forward(self, x):
        x = self.model(x)
        return x
