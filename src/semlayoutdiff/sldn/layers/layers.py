import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import resnet18
from einops import rearrange
from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, num_steps, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            Mish()
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, dim_out)
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h += self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class ResNet18(nn.Module):
    def __init__(self, in_dim=1, out_dim=64):
        super(ResNet18, self).__init__()
        self._feature_extractor = resnet18(weights=None)

        self._feature_extractor.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._feature_extractor.conv1 = torch.nn.Conv2d(
            in_dim,
            64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        self._feature_extractor.fc = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, out_dim)
        )

        # Add a transposed convolution to convert the feature vector into a feature map
        self.deconv = nn.ConvTranspose2d(out_dim, out_dim, kernel_size=120, stride=1)

    def forward(self, X):
        X = X.float()
        out = self._feature_extractor(X)
        return out

class SegmentationUnet(nn.Module):
    """U-Net architecture for segmentation with diffusion and conditioning support."""
    
    def __init__(self, num_classes, dim, num_steps, dim_mults=(1, 2, 4, 8), groups=8, dropout=0.):
        super().__init__()
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Embeddings for different inputs and conditions
        self.embedding = nn.Embedding(num_classes, dim)
        self.floorplan_embedding = nn.Embedding(4, dim)
        self.floorplan_encoder = ResNet18(in_dim=1, out_dim=dim)
        self.room_type_embedding = nn.Embedding(3, dim)
        self.mixed_condition_embedding = nn.Embedding(3, dim)
        self.fc_text = nn.Linear(50, dim)
        
        self.dim = dim
        self.num_classes = num_classes
        self.dropout = nn.Dropout(p=dropout)

        # Time embedding and MLPs for conditioning
        self.time_pos_emb = SinusoidalPosEmb(dim, num_steps=num_steps)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.room_type_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.mixed_condition_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        # U-Net architecture
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                Residual(Rezero(LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                Residual(Rezero(LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Block(dim, dim),
            nn.Conv2d(dim, num_classes, 1)
        )

    def forward(self, time, x, floor_plan=None, room_type=None, text=None, mixed_condition_id=None):
        """Forward pass through the U-Net with optional conditioning."""
        x_shape = x.size()[1:]
        if len(x.size()) == 3:
            x = x.unsqueeze(1)

        B, C, H, W = x.size()
        x = self.embedding(x)

        # Add floor plan conditioning if provided
        if floor_plan is not None:
            floor_plan = self.floorplan_embedding(floor_plan)
            x = x + floor_plan

        assert x.shape == (B, C, H, W, self.dim)
        x = x.permute(0, 1, 4, 2, 3)
        assert x.shape == (B, C, self.dim, H, W)
        x = x.reshape(B, C * self.dim, H, W)

        # Process time embedding and conditions
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        if room_type is not None:
            room_type = self.room_type_embedding(room_type)
            room_type = self.room_type_mlp(room_type)
            t += room_type
            
        if mixed_condition_id is not None:
            mixed_condition_id = self.mixed_condition_embedding(mixed_condition_id)
            mixed_condition_id = self.mixed_condition_mlp(mixed_condition_id)
            t += mixed_condition_id
        
        if text is not None:
            text = self.fc_text(text)
            text_global = text.mean(dim=2).squeeze(1)
            t = t + text_global

        # Encoder path
        h = []
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = self.dropout(x)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # Middle blocks
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # Decoder path
        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x).view(B, self.num_classes, *x_shape)
