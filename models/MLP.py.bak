import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_c=784, out_c=128, num_classes=10, values=None):
        super(MLP, self).__init__()

        if values is not None:
            self.values = values.clone()
        else:
            print('*** Values is None')
        self.w_idx = {}

        self.fc1 = nn.Linear(in_c, out_c, bias=False)
        self.fc2 = nn.Linear(out_c, num_classes, bias=False)

        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        return x

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
                            sigma = grad_abs_flat.std() + 1e-6

                            normal_dist = torch.distributions.Normal(mu, sigma)
                            prob = normal_dist.cdf(grad_abs)
                        elif opt.update_met == 'p0.5':
                            prob = torch.full((grad_abs.size()), 0.5).to(grad.device)
                        elif opt.update_met == 'p0.25':
                            prob = torch.full((grad_abs.size()), 0.25).to(grad.device)
                        else:
                            raise NotImplementedError(opt.update_met)

                        # ====================================================================

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
                            sigma = grad_abs_flat.std() + 1e-6

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
                            sigma = grad_abs_flat.std() + 1e-6

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
