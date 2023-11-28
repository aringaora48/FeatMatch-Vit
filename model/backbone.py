import torch
from torch import nn
import torchvision.models as models
from torch.cuda import amp

from .wide_resnet import WideResNet
from .sslnet import SSLNet
from .Vit import ViT 

def get_model(arch):
    if arch == 'cnn-13':
        model = SSLNet(10)  # 예시로 클래스 수를 10으로 설정
    elif arch.split('-')[0] == 'resnet':
        arch, depth = arch.split('-')
        model = getattr(models, arch+depth)()
    elif arch.split('-')[0] == 'wresnet':
        arch, depth, width = arch.split('-')
        depth, width = int(depth), int(width)
        model = WideResNet(num_classes=10, depth=depth, widen_factor=width)
    elif arch == 'vit':
        # ViT 모델 초기화 파라미터는 예시입니다. 실제 파라미터는 사용 사례에 맞게 조정해야 합니다.
        model = ViT(image_size=256, patch_size=32, num_classes=10, dim=1024, depth=6, heads=8, mlp_dim=2048)
    else:
        raise KeyError("Specified architecture is not supported")
    return model


def make_backbone(backbone):
    model = get_model(backbone)
    
    if isinstance(model, ViT):
        fext_modules = [
            model.to_patch_embedding,
            AddPositionEmbedding(model.pos_embedding),
            model.transformer
        ]
        fext_modules.append(MeanPooling() if model.pool == 'mean' else SelectCLS())
        fext = nn.Sequential(*fext_modules)
        fdim = model.mlp_head.in_features
    else:
        # 다른 모델들의 경우 기존 방식을 사용
        fext = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        fdim = model.fc.in_features

    return fext, fdim


class AmpModel(nn.Module):
    def __init__(self, model, amp=True):
        super(AmpModel, self).__init__()
        self.amp = amp
        self.model = model

    def forward(self, x):
        with amp.autocast(enabled=self.amp):
            return self.model(x)

class AddPositionEmbedding(nn.Module):
    def __init__(self, pos_embedding):
        super().__init__()
        self.pos_embedding = pos_embedding

    def forward(self, x):
        return x + self.pos_embedding[:, :(x.shape[1])]

class MeanPooling(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)

class SelectCLS(nn.Module):
    def forward(self, x):
        return x[:, 0]