# Note that in this ldc_model.py, we assume sign(0) = 1.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.empty_cache()
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
else:
    device = torch.device("cpu")

class BinaryActivation(torch.nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        out_forward = torch.sign(out_forward + 0.5)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


torch.seed()

class FeatureLayer(nn.Module):
    def __init__(self, N, FD, VD, lut_nbit, reduction_method):
        super(FeatureLayer, self).__init__()
        self.num_feature = N
        self.fhv_dimension = FD
        self.vhv_dimension = VD
        self.lut_nbit = lut_nbit
        self.reduction_method = reduction_method
        self.rept = int(FD / VD)
        self.weight = torch.nn.Parameter(torch.empty((self.num_feature, self.fhv_dimension)))
        nn.init.xavier_uniform_(self.weight)
        self.f = torch.zeros_like(self.weight.data).to(device)
        self.b = torch.zeros_like(self.weight.data).to(device)
        self.delta = torch.zeros_like(self.weight.data).to(device)
        self.ema = torch.sign(self.weight.data).to(device)
        self.weight_pre = self.weight.data.clone().to(device)
        self.scaling_factor = 0
        self.mask = nn.Parameter(torch.ones_like(self.weight), requires_grad=False)
        self.binarization = BinaryActivation()
        if self.reduction_method == 'lut':
            self.permute = nn.Parameter(torch.from_numpy(np.random.permutation(N)), requires_grad=False)
            self.lut_weights = nn.ParameterList()
            self.lut_pads = nn.ParameterList() # TODO: module weight
            n = self.num_feature
            while n > lut_nbit:
                num_pad = lut_nbit - (n % lut_nbit) if n % lut_nbit != 0 else 0
                param = nn.Parameter(torch.empty((n + num_pad, self.fhv_dimension)))
                nn.init.xavier_uniform_(param)
                self.lut_weights.append(param)
                self.lut_pads.append(nn.ZeroPad2d((0, 0, 0, num_pad)))
                n = (n + num_pad) // lut_nbit
            # print (self.lut_weights)
            # print (self.lut_pads)

    def forward(self, x):
        x = x.view(-1, self.num_feature, self.vhv_dimension).repeat(1, 1, self.rept)
        if self.reduction_method == 'lut':
            x = x[:, self.permute, :]
        real_weights = self.weight * self.mask
        self.scaling_factor = torch.mean(abs(real_weights) * (1 - self.b))
        self.scaling_factor = self.scaling_factor.detach()
        binary_weights_no_grad = self.scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        tmp = x * binary_weights

        if self.reduction_method == 'lut':
            for pad, lut_weight in zip(self.lut_pads, self.lut_weights):
                tmp = pad(tmp) * lut_weight
                tmp = tmp.view(-1, lut_weight.shape[0] // self.lut_nbit, self.lut_nbit, self.fhv_dimension)
                tmp = torch.sum(tmp, dim=2)
                tmp = self.binarization(tmp)

        y = torch.sum(tmp, dim=1)
        return y
    
    def prune_input(self, prune_level=0.5):
        # Prune based on magnitude of real-valued weights
        threshold = torch.quantile(torch.mean(torch.abs(self.weight), axis=1), prune_level)
        new_mask = torch.unsqueeze((torch.mean(torch.abs(self.weight), axis=1) > threshold).float(), 1)
        self.mask.copy_(new_mask)

    def prune_weight(self, prune_level=0.5):
        # Prune based on magnitude of real-valued weights
        threshold = torch.quantile(torch.abs(self.weight), prune_level)
        new_mask = (torch.abs(self.weight) > threshold).float()
        self.mask.copy_(new_mask)

    def prune_input_and_weight(self, prune_input_level=0.5, prune_weight_level=0.5):
        threshold = torch.quantile(torch.mean(torch.abs(self.weight), axis=1), prune_input_level)
        input_mask = (torch.mean(torch.abs(self.weight), axis=1) > threshold).float()

        threshold = torch.quantile(torch.abs(self.weight[torch.flatten(torch.nonzero(input_mask))]), prune_weight_level)
        weight_mask = (torch.abs(self.weight) > threshold).float()

        new_mask = torch.unsqueeze(input_mask, 1) * weight_mask
        self.mask.copy_(new_mask)


class ClassLayer(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(ClassLayer, self).__init__()
        self.shape = (out_shape, in_shape)
        self.weight = torch.nn.Parameter(torch.empty(self.shape))
        nn.init.xavier_uniform_(self.weight)
        self.f = torch.zeros_like(self.weight.data).to(device)
        self.b = torch.zeros_like(self.weight.data).to(device)
        self.delta = torch.zeros_like(self.weight.data).to(device)
        self.ema = torch.sign(self.weight.data).to(device)
        self.weight_pre = self.weight.data.clone().to(device)
        self.scaling_factor = 0
        self.mask = nn.Parameter(torch.ones_like(self.weight), requires_grad=False)

    def forward(self, x):
        real_weights = self.weight * self.mask
        self.scaling_factor = torch.mean(abs(real_weights) * (1-self.b))
        self.scaling_factor = self.scaling_factor.detach()
        binary_weights_no_grad = self.scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.linear(x, binary_weights)
        return y
    
    def prune_weight(self, prune_level=0.5):
        # Prune based on magnitude of real-valued weights
        threshold = torch.quantile(torch.abs(self.weight), prune_level)
        new_mask = (torch.abs(self.weight) > threshold).float()
        self.mask.copy_(new_mask)

class ValueBox(nn.Module):
    def __init__(self, D):
        super(ValueBox, self).__init__()
        self.dimension = D
        self.fc1 = nn.Linear(1, 20, bias=True)
        self.bn = nn.BatchNorm1d(20)
        self.fc3 = nn.Linear(20, self.dimension, bias = True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        # x = torch.tanh(x)
        x = self.fc3(x)
        return x

class ValFeaClaSearchNet(nn.Module):
    def __init__(self, N, M, FD, VD, K, dp = 0, enable_binarize=True, lut_nbit=6, reduction_method='add'):
        super(ValFeaClaSearchNet, self).__init__()
        self.num_feature = N
        self.num_value = M
        self.fhv_dimension = FD
        self.vhv_dimension = VD
        self.num_class = K
        self.drop_prob = dp
        self.enable_binarize = enable_binarize
        self.fc0 = ValueBox(self.vhv_dimension)
        self.fc1 = FeatureLayer(self.num_feature, self.fhv_dimension, self.vhv_dimension, lut_nbit, reduction_method)
        self.binarization = BinaryActivation()
        self.fc2 = ClassLayer(self.fhv_dimension, self.num_class)

    def forward(self, x):
        if x.shape[1] != self.num_feature:
            raise ValueError('The input shape[1] '+str(x.shape[1])+' does not match '+str(self.num_feature)+'.')
        x = x.reshape(-1, 1)
        x = self.fc0(x)
        x = self.binarization(x.view(-1,self.num_feature*self.vhv_dimension))
        x = self.fc1(x)
        if self.enable_binarize:
            x = self.binarization(x)
        x = self.fc2(x)
        return x

def QAT(model, M=0.025, fth=0.025*3.3):
    cnt = 0
    cnt1 = 0
    with torch.no_grad():
        for mod in model.modules():
            if hasattr(mod, 'b'):
                weight0 = mod.weight_pre.sign()
                weight1 = mod.weight.data.sign()
                delta_pre = mod.delta.clone()
                mod.delta = weight1 - weight0
                o = ((mod.delta.sign() != delta_pre.sign()) * (mod.delta != 0)).int()
                mod.f = M * o + (1 - M) * mod.f
                idxs = (mod.f > fth).int().nonzero(as_tuple=True)
                mod.b[idxs] = 1
                mod.weight.data[idxs] = mod.ema[idxs].sign()
                # mod.weight.data[idxs] = mod.weight_pre[idxs].clone()
                mod.ema = M * mod.weight_pre + (1 - M) * mod.ema
                mod.weight_pre = mod.weight.data.clone()
                cnt += mod.b.numel()
                cnt1 += mod.b.sum()
    return cnt1 / cnt
