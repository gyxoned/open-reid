import torch
import torch.nn as nn

class _AdaptBatchNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, adaptation=True):
        super(_AdaptBatchNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.adaptation = adaptation
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        # self.register_buffer('running_mean', torch.zeros(num_features))
        # self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        # self.running_mean.zero_()
        # self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _calculate_mean_var(self, input):
        c = input.size(1)
        i = input.transpose(1, -1).contiguous()
        i = i.view(-1, c)
        mean = i.mean(0)
        var = i.var(0)
        return mean, var

    def _forward_train(self, s_input, t_input, s_mean, s_var, t_mean, t_var):
        pass

    def _forward_test(self, t_input):
        pass

    def forward(self, inputs):
        if self.adaptation:
            bs = inputs.size(0)
            assert (bs%2==0)
            s_input, t_input = torch.split(inputs, bs//2, dim=0)
            s_mean, s_var = self._calculate_mean_var(s_input)
            t_mean, t_var = self._calculate_mean_var(t_input)
            return self._forward_train(s_input, t_input, s_mean, s_var, t_mean, t_var)
        else:
            return self._forward_test(inputs)


class AdaptBatchNorm1d(_AdaptBatchNorm):

    def _forward_train(self, s_input, t_input, s_mean, s_var, t_mean, t_var):
        s_mean = s_mean.view(1, -1, 1).expand_as(s_input)
        s_var = s_var.view(1, -1, 1).expand_as(s_input)
        t_mean = t_mean.view(1, -1, 1).expand_as(t_input)
        t_var = t_var.view(1, -1, 1).expand_as(t_input)

        out = (s_input-s_mean) / torch.sqrt(s_var+self.eps) * torch.sqrt(t_var+self.eps) + t_mean
        if self.affine:
            out = self.weight.view(1, -1, 1).expand_as(s_input) * out + self.bias.view(1, -1, 1).expand_as(s_input)
        return torch.cat((out, t_input))

    def _forward_test(self, t_input):
        out = t_input
        if self.affine:
            out = self.weight.view(1, -1, 1).expand_as(t_input) * out + self.bias.view(1, -1, 1).expand_as(t_input)
        return out

class AdaptBatchNorm2d(_AdaptBatchNorm):

    def _forward_train(self, s_input, t_input, s_mean, s_var, t_mean, t_var):
        s_mean = s_mean.view(1, -1, 1, 1).expand_as(s_input)
        s_var = s_var.view(1, -1, 1, 1).expand_as(s_input)
        t_mean = t_mean.view(1, -1, 1, 1).expand_as(t_input)
        t_var = t_var.view(1, -1, 1, 1).expand_as(t_input)

        out = (s_input-s_mean) / torch.sqrt(s_var+self.eps) * torch.sqrt(t_var+self.eps) + t_mean
        if self.affine:
            out = self.weight.view(1, -1, 1, 1).expand_as(s_input) * out + self.bias.view(1, -1, 1, 1).expand_as(s_input)
        return torch.cat((out, t_input))

    def _forward_test(self, t_input):
        out = t_input
        if self.affine:
            out = self.weight.view(1, -1, 1, 1).expand_as(t_input) * out + self.bias.view(1, -1, 1, 1).expand_as(t_input)
        return out