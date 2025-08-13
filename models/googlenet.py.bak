import warnings
from collections import namedtuple
from functools import partial
from typing import Any, Callable, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .layers import CustomConv2d


__all__ = ["GoogLeNet", "GoogLeNetOutputs", "_GoogLeNetOutputs", "googlenet"]


GoogLeNetOutputs = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"])
GoogLeNetOutputs.__annotations__ = {"logits": Tensor, "aux_logits2": Optional[Tensor], "aux_logits1": Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _GoogLeNetOutputs set here for backwards compat
_GoogLeNetOutputs = GoogLeNetOutputs


class GoogLeNet(nn.Module):
    __constants__ = ["aux_logits", "transform_input"]

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        transform_input: bool = False,
        init_weights: Optional[bool] = None,
        blocks: Optional[list[Callable[..., nn.Module]]] = None,
        dropout: float = 0.2,
        dropout_aux: float = 0.7,
        values: List[float] = None,
    ) -> None:
        super().__init__()
        if values is not None:
            self.values = values.clone()
        self.w_idx = {}

        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        if init_weights is None:
            warnings.warn(
                "The default weight initialization of GoogleNet will be changed in future releases of "
                "torchvision. If you wish to keep the old behavior (which leads to long initialization times"
                " due to scipy/scipy#11299), please set init_weights=True.",
                FutureWarning,
            )
            init_weights = True
        if len(blocks) != 3:
            raise ValueError(f"blocks length should be 3 instead of {len(blocks)}")
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes, dropout=dropout_aux)
            self.aux2 = inception_aux_block(528, num_classes, dropout=dropout_aux)
        else:
            self.aux1 = None  # type: ignore[assignment]
            self.aux2 = None  # type: ignore[assignment]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes, bias=False)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux1: Optional[Tensor] = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        aux2: Optional[Tensor] = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux2, aux1

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

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux2: Tensor, aux1: Optional[Tensor]) -> GoogLeNetOutputs:
        if self.training and self.aux_logits:
            return _GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x  # type: ignore[return-value]

    def forward(self, x: Tensor) -> GoogLeNetOutputs:
        x = self._transform_input(x)
        x, aux2, aux1 = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted GoogleNet always returns GoogleNetOutputs Tuple")
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return self.eager_outputs(x, aux2, aux1)


class Inception(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1), conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1),
        )

    def _forward(self, x: Tensor) -> list[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.7,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024, bias=False)
        self.fc2 = nn.Linear(1024, num_classes, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = self.dropout(x)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = CustomConv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


def googlenet(*, weights = None, progress: bool = True, **kwargs: Any) -> GoogLeNet:
    """GoogLeNet (Inception v1) model architecture from
    `Going Deeper with Convolutions <http://arxiv.org/abs/1409.4842>`_.

    Args:
        weights (:class:`~torchvision.models.GoogLeNet_Weights`, optional): The
            pretrained weights for the model. See
            :class:`~torchvision.models.GoogLeNet_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.GoogLeNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.GoogLeNet_Weights
        :members:
    """
    weights = None

    original_aux_logits = kwargs.get("aux_logits", False)

    model = GoogLeNet(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        if not original_aux_logits:
            model.aux_logits = False
            model.aux1 = None  # type: ignore[assignment]
            model.aux2 = None  # type: ignore[assignment]
        else:
            warnings.warn(
                "auxiliary heads in the pretrained googlenet model are NOT pretrained, so make sure to train them"
            )

    return model
