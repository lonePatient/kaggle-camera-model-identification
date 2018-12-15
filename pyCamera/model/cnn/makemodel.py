#encoding:utf-8
import torch.nn as nn
from .dpn import dpn92,dpn98,AdaptiveAvgMaxPool2d
from .resnext import resnext101_32x4d
from .senet import se_resnet50,se_resnext50_32x4d,se_resnet101
from torchvision.models import resnet50,resnet101,resnet152
from torchvision.models import densenet201,densenet161,densenet121,densenet169

# 初始化模型结构
class MakeModel(nn.Module):
    def __init__(self, arch, num_classes = 10, pretrained = False):
        super(MakeModel,self).__init__()
        self.num_classes = num_classes
        self.arch = arch
        self.pretrained = pretrained
        self._reset_model()

    # 初始化网络
    # 网络权重位于: ~/.torch/models/
    def _reset_model(self):
        model = None
        # ------------- densenet网络 ---------------------------
        if self.arch.startswith('densenet'):
            if self.arch.startswith('densenet201'):
                model = densenet201(pretrained = self.pretrained)
            elif self.arch.startswith('densenet161'):
                model = densenet161(pretrained = self.pretrained)
            elif self.arch.startswith('densenet121'):
                model = densenet121(pretrained = self.pretrained)
            elif self.arch.startswith('densenet169'):
                model = densenet169(pretrained = self.pretrained)

            num_features = model.classifier.in_features
            self._features = nn.Sequential(
                model.features,
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=7, stride=1)
            )
            self._classifier =nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128,self.num_classes)
            )

        # ------------- resnet网络 ---------------------------
        elif self.arch.startswith('resnet'):
            if self.arch.startswith('resnet50'):
                model = resnet50(pretrained=self.pretrained)
            if self.arch.startswith('resnet101'):
                model = resnet101(pretrained=self.pretrained)
            if self.arch.startswith('resnet152'):
                model = resnet152(pretrained=self.pretrained)

            num_features = model.fc.in_features
            self._features = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
                nn.AdaptiveAvgPool2d(1),
            )
            self._classifier = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128,self.num_classes)
            )

        # ------------- dpn网络 ---------------------------
        elif 'dpn' in self.arch:
            if self.arch.startswith('dpn92'):
                model = dpn92(pretrained='imagenet+5k' if self.pretrained else self.pretrained)
            elif self.arch.startswith('dpn98'):
                model = dpn98(pretrained='imagenet' if self.pretrained else self.pretrained)

            num_features = model.in_chs
            self._features = nn.Sequential(
                model.features,
                AdaptiveAvgMaxPool2d(pool_type='avg')
            )
            self._classifier = nn.Conv2d(num_features,self.num_classes, kernel_size=1, bias=True)

        # ------------- resnext网络 ---------------------------
        elif self.arch.startswith('resnext'):
            if self.arch.startswith('resnext101'):
                model = resnext101_32x4d(pretrained='imagenet' if self.pretrained else None)

            num_features = model.last_linear.in_features
            self._features = nn.Sequential(
                model.features,
                model.avg_pool,

                )
            self._classifier =nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128,self.num_classes)
            )

        # ------------- senet网络 ---------------------------
        elif self.arch.startswith('se_resnet') or self.arch.startswith('se_resnext'):
            if self.arch.startswith('se_resnet50'):
                model = se_resnet50(pretrained='imagenet' if self.pretrained else None)
            if self.arch.startswith('se_resnet101'):
                model = se_resnet101(pretrained='imagenet' if self.pretrained else None)
            if self.arch.startswith('se_resnext50'):
                model = se_resnext50_32x4d(pretrained='imagenet' if self.pretrained else None)

            num_features = model.last_linear.in_features
            self._features = nn.Sequential(
                model.layer0,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
                nn.AdaptiveAvgPool2d(1),
            )
            self._classifier = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, self.num_classes)
            )

        if model is None:
            raise RuntimeError('Unknown model architecture: {}'.format(self.arch))

    def forward(self,inputs):
        x = self._features(inputs)
        if not 'dpn' in self.arch:
            x = x.view(x.size(0),-1)
        out = self._classifier(x)
        if  'dpn' in self.arch:
            out = out.view(out.size(0),-1)
        return out
