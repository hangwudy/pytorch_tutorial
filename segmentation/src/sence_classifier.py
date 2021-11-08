# Scene classification & segmentation

from collections import OrderedDict, namedtuple
from typing import Dict, Optional, List, Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter


class SceneParserModel(nn.Module):
    def __init__(self, backbone: nn.Module, aspp: nn.Module, deeplab: nn.Module,
                 scene_classifier: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.aspp = aspp
        self.deeplab = deeplab
        self.scene_classifier = scene_classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x_out_2 = features["out_2"]
        x_out_3 = features["out_3"]
        x_out_4 = features["out_4"]
        x_aspp = self.aspp(x_out_4)
        print(x_aspp.shape)
        print(x_out_2.shape)
        print(x_out_3.shape)
        print(x_out_4.shape)
        x = self.deeplab(x_aspp)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["seg"] = x

        if self.scene_classifier is not None:
            x_cat = torch.cat((x_aspp, x_out_2, x_out_3, x_out_4), dim=1)
            print(x_cat.shape)
            x = self.scene_classifier(x_cat)
            result["cls"] = x

        return result


class DeepLabHead(nn.Sequential):
    def __init__(self, num_classes: int) -> None:
        super().__init__(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class SceneParserHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SceneParserHead, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, 2048, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.avgpool(x)
        x = torch.flatten(x)
        x = self.fc(x)
        return x


class ModelWrapper(torch.nn.Module):
    """
    Wrapper class for model with dict/list rvalues.
    https://discuss.pytorch.org/t/a-tensorboard-problem-about-use-add-graph-method-for-deeplab-v3-in-torchvision/95808/2
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Init call.
        """
        super().__init__()
        self.model = model

    def forward(self, input_x: torch.Tensor) -> Any:
        """
        Wrap forward call.
        """
        data = self.model(input_x)

        if isinstance(data, dict):
            data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))  # type: ignore
            data = data_named_tuple(**data)  # type: ignore

        elif isinstance(data, list):
            data = tuple(data)

        return data


def build():
    num_classes = 21
    return_layers = {"layer2": 'out_2', "layer3": 'out_3', "layer4": 'out_4'}
    backbone = resnet.resnet50(
        pretrained=True,
        replace_stride_with_dilation=[False, True, True])
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    aspp = ASPP(2048, [12, 24, 36])
    deeplab_head = DeepLabHead(num_classes)
    # ASPP: 256, Feature 2: 512, Feature 3: 1024, Feature 4: 2048
    scene_classifier = SceneParserHead((256 + 512 + 1024 + 2048), 2)
    model = SceneParserModel(backbone=backbone, aspp=aspp, deeplab=deeplab_head, scene_classifier=scene_classifier)
    print(model)
    inp = torch.randn((1, 3, 640, 480))
    print(inp.shape)
    # export graph
    writer = SummaryWriter('logs')
    model_wrapper = ModelWrapper(model)
    writer.add_graph(model_wrapper, inp)
    writer.close()

    # test run
    model.eval()
    out = model(inp)
    print(out)


if __name__ == '__main__':
    build()
