"""
以下是基于原论文tensorflow的复现，大体框架都是按照原论文来的，参照了Niels Rogge and Kashif Rasul的The Annotated Diffusion Model
"""



import math
from inspect import isfunction
from functools import partial # partial用于固定部分参数


from tqdm.auto import tqdm # tqdm.auto中的tqdm自动选择合适当前环境的进度条实现，例如终端/Jupyter Notebook
from einops import rearrange, reduce # rearrange重新排列张量维度，reduce沿指定维度进行聚合计算,两者都是函数
from einops.layers.torch import Rearrange # Rearrange是nn.Module，相当于一个网络层

import torch
from torch import nn, einsum # einsum可以用一条表达式完成矩阵乘法、张量求和、维度变换等操作
from torch.nn import functional as F

"""Network helpers"""

def exists(x):
    return x is not None

def default(val, d):
    """
    val是None时调用d
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d

def num_to_groups(num, division):
    groups = num // division
    remainder = num % division
    arr = [division] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), kernel_size=3, padding=1)
    )

def Downsample(dim, dim_out = None):
    """
    Rearrange:(B, C, H, W) -> (B, 4C, H/2, W/2)，空间尺寸减半，通道数量扩大四倍
    1*1卷积不改变空间尺寸
    """
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

"""position embedding"""

class SinusoidalPositionEmbeddings(nn.Module):
    """
    transformer里的PE
    the authors employ sinusoidal position embeddings to encode t
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

"""ResNet Block"""

class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1",  partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class Block(nn.Module):
    """
    卷积块的实现
    采用WeightStandardizedConv2d + GroupNorm + Affine（仿射）+ SiLU
    这个仿射是为了noise predictor中的输入时间步（噪声强度），scale(t), shift(t)，让共享U-net能够处理噪声强度不同的输入
    """
    def __init__(self, dim, dim_out = None, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, default(dim_out, dim), 3, padding = 1)
        self.norm = nn.GroupNorm(groups, default(dim_out, dim))
        self.act = nn.SiLU()
    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (1 + scale) + shift
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        """
        :param time_emb_dim: 时间嵌入维度
        如果不为None，将时间嵌入转化为两组参数scale和shift，供后面的Block调制
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    def forward(self, x, time_emb = None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)
        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

"""Attention module"""

class Attention(nn.Module):
    """
    多头自注意力机制，和Transformer不同处在于处理的是image，需要将图像的高度和宽度合并为一个序列维度
    """
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = heads * dim_head
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(
            lambda t : rearrange(t, 'b (h d) x y -> b h d (x y)', h = self.heads), qkv
        ) # 将二维图片转化为一维处理
        q = q * self.scale
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim = -1, keepdim=True).detach() # 防止softmax过饱和
        attn = sim.softmax(dim = -1)
        out = einsum("b h i j, b h d j -> b h d i", attn, v)
        out = rearrange(out, "b h d (x y) -> b (h d) x y", x = h, y = w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    """
    线性多头注意力，先计算KV
    """
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        hidden_dim = heads * dim_head
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(
            lambda t : rearrange(t, "b (h c) x y -> b h c (x y)", h = self.heads), qkv
        )
        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)
        q = q * self.scale
        context = einsum("b h d n, b h e n -> b h d e", k, v)
        out = einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", x = h, y = w)
        return self.to_out(out)

"""Group Normalization"""

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Unet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim = None,
            out_dim = None,
            dim_mults = (1, 2, 4, 8),
            channels = 3,
            self_condition = False,
            resnet_block_groups = 4,
    ):
        super().__init__()

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding = 0)
        """
        高效生成输入维度和输出维度
        """
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_class = partial(ResnetBlock, groups = resnet_block_groups)

        # time embedding
        time_dim = dim * 4 # 此处的4是超参数，可以自行调整
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_class(dim_in, dim_in, time_emb_dim = time_dim),
                        block_class(dim_in, dim_in, time_emb_dim = time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_class(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_class(mid_dim, mid_dim, time_emb_dim = time_dim)

        for idx, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = idx == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_class(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                        block_class(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding = 1)
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)
        self.final_res_block = block_class(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda : torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
        x = torch.cat((x, r), dim = 1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)

"""beta_schedule"""
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps)**2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos((x / timesteps) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

timesteps = 200

#define beta schedule
betas = linear_beta_schedule(timesteps = timesteps)

#define alpha
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim = 0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.0) # 在alphas前面添加一个1，表示alpha0 = 1，初始状态
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculate for diffusion q(x_{t} | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculate for diffusion q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    """
    每张图片的t都不一样，所以需要进行提取
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

"""image operation"""

from PIL import Image
import requests
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream = True).raw)
# image.show()

import numpy as np
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize, RandomHorizontalFlip

image_size = 128
transform = Compose([
    Resize(image_size),
    CenterCrop(image_size),
    ToTensor(),
    Lambda(lambda t : t * 2 - 1),
])

reverse_transform = Compose([
    Lambda(lambda t : (t + 1) / 2),
    Lambda(lambda t : t.permute(1, 2, 0)),
    Lambda(lambda t : t * 255.),
    Lambda(lambda t : t.numpy().astype(np.uint8)),
    ToPILImage(),
])

x_start = transform(image).unsqueeze(0)
# x_restore = reverse_transform(x_start.squeeze())
# x_restore.show() # 当然分辨率和尺寸会有变化

"""add noise"""

def q_sample(x_start, t, noise = None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def get_noisy_image(x_start, t):
    # add noise
    x_noise = q_sample(x_start, t)

    # turn back into PIL image
    noisy_image = reverse_transform(x_noise.squeeze())
    noisy_image.show()

# take timestep
# t = torch.tensor([40])
# get_noisy_image(x_start, t)

"""loss function"""

def p_losses(denoise_model, x_start, t, noise = None, loss_type = "l1"):
    if noise is None:
        noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start = x_start, t = t, noise = noise)
    predicted_noise = denoise_model(x_noisy, t)
    if loss_type == "l1":
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == "l2":
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()
    return loss


from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms

dataset = load_dataset("fashion_mnist")
image_size = 28
channels = 1
batch_size = 128

transform = Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    Lambda(lambda t : 2 * t - 1)
])

def transforms(examples):
   examples["pixel_values"] = [transform(image.convert("L")) for image in examples["image"]]
   del examples["image"]

   return examples
transformed_dataset = dataset.with_transform(transforms).remove_columns("label")
dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device
    b = shape[0]
    img = torch.randn(shape, device = device)
    imgs = []
    for i in tqdm(reversed(range(0, timesteps), desc = "sampling loop time step", total = timesteps)):
        img = p_sample(model, img, torch.full((b,), i, device = device, dtype = torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

from pathlib import Path
results_folder = Path('./results')
results_folder.mkdir(exist_ok = True)
save_and_sample_every = 1000

from torch.optim import AdamW

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Unet(
    dim = image_size,
    channels = channels,
    dim_mults=(1,2,4)
)
model.to(device)
optimizer = AdamW(model.parameters(), lr = 1e-3)

from torchvision.utils import save_image

epochs = 6

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()

      batch_size = batch["pixel_values"].shape[0]
      batch = batch["pixel_values"].to(device)

      t = torch.randint(0, timesteps, (batch_size,), device=device).long()

      loss = p_losses(model, batch, t, loss_type="huber")

      if step % 100 == 0:
        print("Loss:", loss.item())

      loss.backward()
      optimizer.step()

      # save generated images
      if step != 0 and step % save_and_sample_every == 0:
        milestone = step // save_and_sample_every
        batches = num_to_groups(4, batch_size)
        all_images_list = list(map(lambda n: sample(model, batch_size=n, channels=channels), batches))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)
