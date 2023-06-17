import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np
from unet import UNet

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial
from torch import nn, einsum

import ase
import yaml
import time
import copy
import joblib
import pickle
import numpy as np
import datetime
from ase import Atoms, io, build


CONFIG = {}
CONFIG['data_x_path'] = '/content/perov_5_raw_train_dist_mat.pt'
CONFIG['data_ead_path'] = '/content/perov_5_raw_train_ead_mat.pt'
CONFIG['composition_path'] = '/content/perov_5_raw_train_composition_mat.pt'
CONFIG['cell_path'] = '/content/perov_5_raw_train_cell_mat.pt'
CONFIG['data_y_path'] = "/content/targets.csv"
CONFIG['unprocessed_path'] = '/content/perov_5_raw_train_unprocessed.txt'

CONFIG['data_x_path']

CONFIG['data_x_path'] = 'MP_data_csv/perov_5_raw_train_dist_mat.pt'
CONFIG['data_ead_path'] = 'MP_data_csv/perov_5_raw_train_ead_mat.pt'
CONFIG['composition_path'] = 'MP_data_csv/perov_5_raw_train_composition_mat.pt'
CONFIG['cell_path'] = 'MP_data_csv/perov_5_raw_train_cell_mat.pt'
CONFIG['data_y_path'] = "MP_data_csv/perov_5/raw_train/targets.csv"

CONFIG['unprocessed_path'] = 'MP_data_csv/perov_5_raw_train_unprocessed.txt'

CONFIG['model_path'] = 'saved_models/diff_1d.pt'
CONFIG['scaler_path'] = 'saved_models/diff_scaler.gz'

unprocessed = set()
with open(CONFIG['unprocessed_path'], 'r') as f:
    for l in f.readlines():
        unprocessed.add(int(l))


dist_mat = torch.load(CONFIG['data_x_path'] ,map_location=torch.device('cpu')).to("cpu")
ead_mat = torch.load(CONFIG['data_ead_path'],map_location=torch.device('cpu')).to("cpu")
composition_mat = torch.load(CONFIG['composition_path'],map_location=torch.device('cpu')).to("cpu")
cell_mat = torch.load(CONFIG['cell_path'],map_location=torch.device('cpu')).to("cpu")

# build index
_ind = [i for i in range(dist_mat.shape[0]) if i not in unprocessed]
indices = torch.tensor(_ind, dtype=torch.long).to("cpu")

# select rows torch.Size([27136, 1])
dist_mat = dist_mat[indices] # the torch.load needs the index in tensor format to convert the loaded file in a tensor.
ead_mat = ead_mat[indices]
composition_mat = composition_mat[indices]
cell_mat = cell_mat[indices]

# normalize composition
sums = torch.sum(composition_mat, axis=1).view(-1,1)
composition_mat = composition_mat / sums
composition_mat = torch.cat((composition_mat, sums), dim=1)

y = []
with open(CONFIG['data_y_path'], 'r') as f:
    for i, d in enumerate(f.readlines()):
        if i not in unprocessed:
            y.append(float(d.split(',')[1]))

data_y = np.reshape(np.array(y), (-1,1))

data_y = torch.from_numpy(data_y)
data_y = data_y.to(torch.float32)

data_x = torch.cat((ead_mat/1000000, dist_mat, cell_mat, composition_mat, data_y), dim=1)

mask = data_x[:, 600] <= 10
data_x = data_x[mask]
#data_x, data_y = data_x[:, 0:708] , data_x[:,708]
data_x, composition_mat, data_y = data_x[:, 0:607], data_x[:,607:708] , data_x[:,708]

scaler = MinMaxScaler()
scaler.fit(data_x)
data_x = scaler.transform(data_x)
joblib.dump(scaler, CONFIG['scaler_path']) # save the scaler to be used for later purpose on testing data.

comp1, comp2 = composition_mat[:, 0:96], composition_mat[:,100]/5
comp1 = (comp1.to(torch.float32))
comp2 = comp2.to(torch.float32).view(-1,1)

composition_mat_add = torch.cat((comp1,comp2), dim=1)

data_x = torch.from_numpy(data_x)
data_x = data_x.to(torch.float32)

#composition_mat = torch.from_numpy(composition_mat_add)
#composition_mat = composition_mat.to(torch.float32)

data_x = torch.cat((data_x,composition_mat_add), dim=1)

data_y = data_y.view(-1,1)

data_x = data_x[0:100,:]
data_y = data_y[0:100,:]

print(data_x.shape)
print(data_y.shape)

print(data_x[0])

CONFIG['seed'] = 42
CONFIG['split_ratio'] = 0.2
batch_size = 2


# train/test split and create torch dataloader
xtrain, xtest, ytrain, ytest = train_test_split(data_x, data_y, test_size=CONFIG['split_ratio'], random_state= CONFIG['seed'])

if not isinstance(xtrain, torch.Tensor):
    x_train = torch.tensor(xtrain, dtype=torch.float)
else:
    x_train = xtrain

if not isinstance(ytrain, torch.Tensor):
    y_train = torch.tensor(ytrain, dtype=torch.float)
else:
    y_train = ytrain

if not isinstance(xtest, torch.Tensor):
    x_test = torch.tensor(xtest, dtype=torch.float)
else:
    x_test = xtest

if not isinstance(ytest, torch.Tensor):
    y_test = torch.tensor(ytest, dtype=torch.float)
else:
    y_test = ytest


indices = ~torch.any(x_train.isnan(),dim=1)

x_train = x_train[indices]
y_train = y_train[indices] # y_train is the condition

indices = ~torch.any(x_train[:,:601] > 10 ,dim=1)
x_train = x_train[indices]
y_train = y_train[indices]

indices = ~torch.any(x_test.isnan(),dim=1)
print(indices) # tensor([True, True, True,  ..., True, True, True])

x_test = x_test[indices]
y_test = y_test[indices]
indices = ~torch.any(x_test[:,:601] > 10 ,dim=1)
x_test = x_test[indices]
y_test = y_test[indices]

train_loader = DataLoader(
    TensorDataset(x_train, y_train),
    batch_size=batch_size, shuffle=True, drop_last=False
)

test_loader = DataLoader(
    TensorDataset(x_test, y_test),
    batch_size=batch_size, shuffle=False, drop_last=False
)   

# --------- -------- ------------ ------------
# Model is similar to lucidrain's Diffusion model implimentation.
# --------- -------- ------------ ------------

class DiffusionModel:
    def __init__(self, start_schedule=0.0001, end_schedule=0.02, timesteps = 500):
        self.start_schedule = start_schedule
        self.end_schedule = end_schedule
        self.timesteps = timesteps
        self.betas = torch.linspace(start_schedule, end_schedule, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def forward(self, x_0, t, device):
        #noise = torch.randn(200)
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.alphas_cumprod.sqrt(), t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x_0.shape)

        mean = sqrt_alphas_cumprod_t.to(device) * x_0.to(device)
        variance = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)

        return mean + variance, noise.to(device)

    @torch.no_grad()
    def backward(self, x, t, model, **kwargs):
        """
        Calls the model to predict the noise in the image and returns
        the denoised image.
        Applies noise to this image, if we are not in the last step yet.
        """

        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(torch.sqrt(1.0 / self.alphas), t, x.shape)
        mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t, **kwargs)[0] / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = betas_t

        if t == 0:
            return mean
        else:
            noise = torch.randn_like(x)
            variance = torch.sqrt(posterior_variance_t) * noise
            return mean + variance

    @staticmethod
    def get_index_from_list(values, t, x_shape):
        batch_size = t.shape[0]
        result = values.gather(-1, t.cpu())

        return result.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def kl_divergence(z, mu, std):
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)
    kl = kl.sum(-1)
    return kl

class SinusoidalPositionEmbeddings(nn.Module):
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

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

# model
class Unet1D(nn.Module):
    def __init__(
        self,
        dim,
        inp_dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 1,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]

        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

        self.mu = nn.Linear(inp_dim, inp_dim)
        self.var = nn.Linear(inp_dim, inp_dim)

    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = x.unsqueeze(1)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

            #print('downsample',downsample)
            #print(x.shape)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:

            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

            #print('upsample',upsample)
            #print(x.shape)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)

        x = self.final_conv(x)

        x = x.squeeze()

        z_mu = self.mu(x)
        z_var = self.var(x)

        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        z = x_sample #torch.cat((x_sample,y), dim=1)

        return z, z_mu, z_var

def plot_noise_distribution(noise, predicted_noise):
    plt.hist(noise.cpu().numpy().flatten(), density = True, alpha = 0.8, label = "ground truth noise")
    plt.hist(predicted_noise.cpu().numpy().flatten(), density = True, alpha = 0.8, label = "predicted noise")
    plt.legend()
    plt.show()

# ------- --------- -------- -----------

BATCH_SIZE = 2
NO_EPOCHS = 100
PRINT_FREQUENCY = 10
LR = 0.0001
VERBOSE = True
device = 'cpu'

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# ------- --------- -------- -----------

diffusion_model = DiffusionModel()
unet = Unet1D(64,704)
unet.to(device)
optimizer = torch.optim.Adam(unet.parameters(), lr=LR)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = unet
num_params = count_parameters(model)
print("Number of parameters: {:,}".format(num_params))

batch_size = 2
input_size = 704
kernel_size = 7
padding = 3
output_size = 1  # Desired number of output channels

# ------- --------- -------- -----------
# 3-2-1 Ho jaye shuru
# ------- --------- -------- -----------

print('trainer is running wild')

for epoch in range(100):

    mean_epoch_loss = []
    mean_epoch_loss_val = []
    for batch in train_loader:

        batch = batch[0]

        t = torch.randint(0, diffusion_model.timesteps, (BATCH_SIZE,)).long().to(device)
        try:
          batch_noisy, noise = diffusion_model.forward(batch, t, device)
        except:
          continue

        z , z_mu, z_var = unet(batch_noisy, t)
        predicted_noise = z

        optimizer.zero_grad()
        kld = torch.mean(-0.5 * torch.sum(1 + z_var - z_mu ** 2 - z_var.exp(), dim = 1), dim = 0)
        loss = kld 
        mean_epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    for batch in test_loader:

        batch = batch[0]
        t = torch.randint(0, diffusion_model.timesteps, (BATCH_SIZE,)).long().to(device)

        try:
          batch_noisy, noise = diffusion_model.forward(batch, t, device)
        except:
          continue

        z , z_mu, z_var = unet(batch_noisy, t)
        predicted_noise = z

        optimizer.zero_grad()
        kld = torch.mean(-0.5 * torch.sum(1 + z_var - z_mu ** 2 - z_var.exp(), dim = 1), dim = 0)
        loss = kld
        mean_epoch_loss_val.append(loss.item())

    if epoch % PRINT_FREQUENCY == 0:
        print('---')
        print(f"Epoch: {epoch} | Train Loss {np.mean(mean_epoch_loss)} | Val Loss {np.mean(mean_epoch_loss_val)}")
        #if VERBOSE:
        #    with torch.no_grad():
        #        plot_noise_distribution(noise, predicted_noise)

        torch.save(unet.state_dict(), f"epoch: {epoch}")

# ------- --------- -------- ----------

with torch.no_grad():
    img = torch.randn(1,704).to(device)
    for i in reversed(range(diffusion_model.timesteps)):
        t = torch.full((1,), i, dtype=torch.long, device=device)
        img = diffusion_model.backward(img, t, unet.eval())
        if i % 250 == 0:
            print('i',i)
            print(img)


