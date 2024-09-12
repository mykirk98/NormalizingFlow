import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
# from debug.debugger import *
seed = torch.randint(0, 10000, (1,)).item()
torch.random.manual_seed(seed)

logabs = lambda x: torch.log(torch.abs(x))

class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc

class ChannelPermute(nn.Module):
    def __init__(self, num_channels):
        super(ChannelPermute, self).__init__()
        # Create as many permutations as num_channels and directly register them as a buffer
        self.register_buffer('permutation', torch.randperm(num_channels))
        self.register_buffer('inverse_permutation', torch.argsort(self.permutation))

    def forward(self, x):
        return x[:, self.permutation, :, :]

    def reverse(self, x):
        return x[:, self.inverse_permutation, :, :]


class AffineCoupling(nn.Module):
    def __init__(self, in_channels, hidden_ratio, kernel_size):
        super(AffineCoupling, self).__init__()

        hidden_channels = int(in_channels * hidden_ratio)
        padding = int(kernel_size // 2)
        self.conv1 = nn.Conv2d(int(in_channels), hidden_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(hidden_channels, int(in_channels * 2), kernel_size, padding=padding)


    def forward(self, x):
        hs = F.relu(self.conv1(x))
        out_s = self.conv2(hs)

        scale, translation = torch.chunk(out_s, 2, dim=1)

        # Store the values
        self.last_scale = scale
        self.last_translation = translation

        return x, scale, translation


    def inverse(self, y):
        # Use the stored values
        if hasattr(self, 'last_scales') and hasattr(self, 'last_translations'):
            scales = self.last_scales
            translations = self.last_translations

            x = (y - translations) * torch.exp(-scales)
            return x
        else:
            raise ValueError("No scales and translations stored from forward pass.")



class DoubleFlow(nn.Module):
    def __init__(self, in_channels, hidden_ratio, kernel_size):
        super(DoubleFlow, self).__init__()

        self.clamp = 1.
        self.kernel_size = kernel_size

        half_channels = in_channels // 2
        self.layer1 = AffineCoupling(half_channels, hidden_ratio, kernel_size)
        self.layer2 = AffineCoupling(half_channels, hidden_ratio, kernel_size)

        self.permute = ChannelPermute(in_channels)
        self.actnorm = ActNorm(in_channels)

    def forward(self, x):
        x = self.permute(x)

        x, log_det_actnorm = self.actnorm(x)
        
        x1, x2 = torch.chunk(x, 2, dim=1)
        
        _, scales1, translations1 = self.layer1(x1)
        
        exp_scales1 = self.e(scales1)
        
        y1 = x2 * exp_scales1 + translations1

        _, scales2, translations2 = self.layer2(y1)
        
        exp_scales2 = self.e(scales2)
        
        y2 = x1 * exp_scales2 + translations2

        y = torch.cat([y1, y2], dim=1)
        log_det = log_det_actnorm + torch.log(exp_scales1).sum(dim=[1, 2, 3]) + torch.log(exp_scales2).sum(dim=[1, 2, 3])
        
        return y, log_det
    

    
    def e(self, s):
        return torch.exp(self.log_e(s))
    
    
    
    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    
    def reverse(self, y):
        # Reverse the concatenation
        y1, y2 = torch.chunk(y, 2, dim=1)

        # Reverse the second affine coupling layer
        _, inv_scales2, inv_translations2 = self.layer2.inverse(y1)
        x1 = (y2 - inv_translations2) * torch.exp(-inv_scales2)

        # Reverse the first affine coupling layer
        _, inv_scales1, inv_translations1 = self.layer1.inverse(x1)
        x2 = (y1 - inv_translations1) * torch.exp(-inv_scales1)

        # Reverse the permutation
        x = self.permute.inverse(torch.cat([x1, x2], dim=1))

        # Reverse the actnorm
        x = self.actnorm.inverse(x)

        return x


class MultiFlow(nn.Module):
    def __init__(self, in_channels, hidden_ratio):
        super(MultiFlow, self).__init__()
        
        self.flow1 = DoubleFlow(int(in_channels // 2), hidden_ratio, kernel_size=3)
        self.flow2 = DoubleFlow(int(in_channels // 2), hidden_ratio, kernel_size=3)
        
        self.flow1_ = DoubleFlow(int(in_channels // 2), hidden_ratio, kernel_size=3)
        self.flow2_ = DoubleFlow(int(in_channels // 2), hidden_ratio, kernel_size=3)
    

    
    def forward(self, x):
        x1, x1_ = torch.chunk(x, 2, dim=1)
        
        x2, log_det2 = self.flow1(x1)
        x2_, log_det2_ = self.flow1_(x1_)
        
        x3, log_det3 = self.flow2(x2)
        x3_, log_det3_ = self.flow2_(x2_)
        
        log_det = log_det2 + log_det2_ + log_det3 + log_det3_
        
        x = torch.cat([x3, x3_], dim=1)
        
        return x, log_det

class Model(nn.Module):
    def __init__(self, eval = False):
        super(Model, self).__init__()
        self.training = not eval
        self.feature_extractor = timm.create_model(
            'vgg11_bn',
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3],
        )
        self.flows = nn.ModuleList([
            MultiFlow(128, 1),
            MultiFlow(256, 1),
            MultiFlow(512, 1)
            ])
        
        channels = [128, 256, 512]
        scales = [4,8,16]
        input_size = 256

        self.norms = nn.ModuleList()
        for channel, scale in zip(channels, scales):
            self.norms.append(
                nn.LayerNorm(
                    [channel, int(input_size / scale), int(input_size / scale)],
                    elementwise_affine=True,
                )
            )
        
        self.max_pools = nn.ModuleList()
        for _ in range(3):
            self.max_pools.append(nn.MaxPool2d(2, 2))
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    
    def forward(self, x):
        loss = 0
        outputs = []

        features = self.feature_extractor(x)

        for f, mp, norm, flow in zip(features, self.max_pools, self.norms, self.flows):
            f = mp(f)
            f = norm(f)
            y, log_det = flow(f)
            # 출력 feature map이 표준 정규 분포처럼 보이도록 만들기 위해
            # volume의 과도한 변형을 지양하기위해 log_det 빼줌
            loss += torch.mean(0.5 * torch.sum(y ** 2, dim=(1, 2, 3)) - log_det)
            outputs.append(y)

        ret = {"loss": loss}

        if  not self.training:
            anomaly_map_list = [
                F.interpolate(
                    -torch.exp(-0.5 * torch.mean(output**2, dim=1, keepdim=True)),
                    size=x.shape[2:],
                    mode="bilinear",
                    align_corners=False
                ) for output in outputs
            ]
            ret["anomaly_map"] = torch.mean(torch.stack(anomaly_map_list, dim=-1), dim=-1)

        return ret



if __name__ == "__main__":
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = Model()
    model = model.to(dev)
    model.eval()
    input_tensor = torch.ones((1, 3, 256, 256)).to(dev)
    with torch.no_grad():
        _ = model(input_tensor)

    from torchinfo import summary
    summary(model, input_size=(1, 3, 256, 256), device='cpu')