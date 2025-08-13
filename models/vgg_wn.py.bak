from typing import Callable, List, Optional, Type, Union, Dict, cast
from .layers import CustomConv2d

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return CustomConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return CustomConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class VGG(nn.Module):
    def __init__(self, vgg_cfg: List[Union[str, int]], batch_norm: bool = False, values: List[float] = None,  num_classes: int = 1000) -> None:
        super(VGG, self).__init__()
        self.features = _make_layers(vgg_cfg, batch_norm)
        if values is not None:
            self.values = values.clone()
        else:
            print('*** Values is None')
        self.w_idx = {}
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096, bias=False),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096, bias=False),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes, bias=False),
        )

    def initialize_indexes(self):
        ones = torch.argmin(torch.abs(self.values - 1)).item()
        zeros = torch.argmin(torch.abs(self.values - 0)).item()
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                probs = torch.exp(-torch.abs(torch.arange(len(self.values)) - zeros).float())
                probs /= probs.sum()

                self.w_idx[name] = torch.multinomial(probs, module.weight.numel(), replacement=True).reshape(
                    module.weight.shape)
                module.weight.data.copy_(self.values[self.w_idx[name]])

            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                self.w_idx[name] = {
                    'weight': torch.full((module.weight.shape), ones),
                    'bias': torch.full((module.bias.shape), zeros),
                }
                module.weight.data.copy_(self.values[self.w_idx[name]['weight']])
                module.bias.data.copy_(self.values[self.w_idx[name]['bias']])

    def index_step(self, opt):
        for name, module in self.named_modules():
            if name in self.w_idx:
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    grad = module.weight.grad
                    grad_abs = torch.abs(module.weight.grad)
                    if grad is not None:
                        if opt.update_met == 'min_max':
                            prob = (grad_abs - grad_abs.min()) / (grad_abs.max() - grad_abs.min())
                        elif opt.update_met == 'cdf':
                            grad_abs_flat = grad_abs.flatten()
                            mu = grad_abs_flat.mean()
                            sigma = grad_abs_flat.std()

                            normal_dist = torch.distributions.Normal(mu, sigma)
                            prob = normal_dist.cdf(grad_abs)
                        elif opt.update_met == 'p0.5':
                            prob = torch.full((grad_abs.size()), 0.5).to(grad.device)
                        elif opt.update_met == 'p0.25':
                            prob = torch.full((grad_abs.size()), 0.25).to(grad.device)
                        elif opt.update_met == 'p0.75':
                            prob = torch.full((grad_abs.size()), 0.75).to(grad.device)
                        elif opt.update_met == 'discrete':
                            prob = torch.full((grad_abs.size()), 1.0).to(grad.device)
                        else:
                            raise NotImplementedError(opt.update_met)

                        random_prob = torch.rand_like(prob)

                        positive_mask = (grad > 0) & (random_prob < prob)
                        negative_mask = (grad < 0) & (random_prob < prob)

                        updated_idx = self.w_idx[name].to(grad.device)
                        updated_idx[positive_mask] -= 1
                        updated_idx[negative_mask] += 1

                        self.w_idx[name] = torch.clamp_(updated_idx, 0, len(self.values) - 1)
                        module.weight.data.copy_(self.values.to(grad.device)[updated_idx])

                elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                    grad_weight = module.weight.grad
                    grad_weight_abs = torch.abs(module.weight.grad)
                    grad_bias = module.bias.grad
                    grad_bias_abs = torch.abs(module.bias.grad)

                    if grad_weight is not None:
                        if opt.update_met == 'min_max':
                            prob_weight = (grad_weight_abs - grad_weight_abs.min()) / (grad_weight_abs.max() - grad_weight_abs.min())
                        elif opt.update_met == 'cdf':
                            grad_abs_flat = grad_weight_abs.flatten()
                            mu = grad_abs_flat.mean()
                            sigma = grad_abs_flat.std()

                            normal_dist = torch.distributions.Normal(mu, sigma)
                            prob_weight = normal_dist.cdf(grad_weight_abs)
                        elif opt.update_met == 'p0.5':
                            prob_weight = torch.full((grad_weight_abs.size()), 0.5).to(grad_weight.device)
                        elif opt.update_met == 'p0.25':
                            prob_weight = torch.full((grad_weight_abs.size()), 0.25).to(grad_weight.device)
                        elif opt.update_met == 'p0.75':
                            prob_weight = torch.full((grad_weight_abs.size()), 0.75).to(grad_weight.device)
                        elif opt.update_met == 'discrete':
                            prob_weight = torch.full((grad_weight_abs.size()), 1.0).to(grad_weight.device)
                        else:
                            raise NotImplementedError(opt.update_met)

                        random_prob_weight = torch.rand_like(prob_weight)

                        positive_mask_weight = (grad_weight > 0) & (random_prob_weight < prob_weight)
                        negative_mask_weight = (grad_weight < 0) & (random_prob_weight < prob_weight)

                        updated_idx_weight = self.w_idx[name]['weight'].to(grad_weight.device)
                        updated_idx_weight[positive_mask_weight] -= 1
                        updated_idx_weight[negative_mask_weight] += 1

                        self.w_idx[name]['weight'] = torch.clamp_(updated_idx_weight, 0, len(self.values) - 1)
                        module.weight.data.copy_(self.values.to(grad_weight)[updated_idx_weight])

                    if grad_bias is not None:
                        if opt.update_met == 'min_max':
                            prob_bias = (grad_bias_abs - grad_bias_abs.min()) / (
                                        grad_bias_abs.max() - grad_bias_abs.min())
                        elif opt.update_met == 'cdf':
                            grad_abs_flat = grad_bias_abs.flatten()
                            mu = grad_abs_flat.mean()
                            sigma = grad_abs_flat.std()

                            normal_dist = torch.distributions.Normal(mu, sigma)
                            prob_bias = normal_dist.cdf(grad_bias_abs)
                        elif opt.update_met == 'p0.5':
                            prob_bias = torch.full((grad_bias_abs.size()), 0.5).to(grad_bias.device)
                        elif opt.update_met == 'p0.25':
                            prob_bias = torch.full((grad_bias_abs.size()), 0.25).to(grad_bias.device)
                        elif opt.update_met == 'p0.75':
                            prob_bias = torch.full((grad_bias_abs.size()), 0.75).to(grad_bias.device)
                        elif opt.update_met == 'discrete':
                            prob_bias = torch.full((grad_bias_abs.size()), 1.).to(grad_bias.device)
                        else:
                            raise NotImplementedError(opt.update_met)

                        random_prob_bias = torch.rand_like(prob_bias)

                        positive_mask_bias = (grad_bias > 0) & (random_prob_bias < prob_bias)
                        negative_mask_bias = (grad_bias < 0) & (random_prob_bias < prob_bias)

                        updated_idx_bias = self.w_idx[name]['bias'].to(grad_bias.device)
                        updated_idx_bias[positive_mask_bias] -= 1
                        updated_idx_bias[negative_mask_bias] += 1

                        self.w_idx[name]['bias'] = torch.clamp_(updated_idx_bias, 0, len(self.values) - 1)
                        module.bias.data.copy_(self.values.to(grad_bias.device)[updated_idx_bias])


    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


def _make_layers(vgg_cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: nn.Sequential[nn.Module] = nn.Sequential()
    in_channels = 3
    for v in vgg_cfg:
        if v == "M":
            layers.append(nn.MaxPool2d((2, 2), (2, 2)))
        else:
            v = cast(int, v)
            conv2d = conv3x3(in_channels, v)
            if batch_norm:
                layers.append(conv2d)
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(True))
            else:
                layers.append(conv2d)
                layers.append(nn.ReLU(True))
            in_channels = v

    return layers

__all__ = [
    "VGG",
    "vgg11", "vgg13", "vgg16", "vgg19",
    "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
]
vgg_cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

def vgg11(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg11"], False, **kwargs)

    return model


def vgg13(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg13"], False, **kwargs)

    return model


def vgg16(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg16"], False, **kwargs)

    return model


def vgg19(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg19"], False, **kwargs)

    return model


def vgg11_bn(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg11"], True, **kwargs)

    return model


def vgg13_bn(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg13"], True, **kwargs)

    return model


def vgg16_bn(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg16"], True, **kwargs)

    return model


def vgg19_bn(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg19"], True, **kwargs)

    return model
