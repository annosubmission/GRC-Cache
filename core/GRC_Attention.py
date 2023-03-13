import torch.nn as nn
import torch
import torch.nn.functional as f
from timm.models.layers import trunc_normal_

# Cached Transformers: Improving Vision Transformers with Differentiable Memory Cache 

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,
                 window_size=None,
                 num_heads=None,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # rel pos
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))

        #
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, kv, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = q.shape
        kv = self.kv(kv).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q(q).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B H N C

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., inner_dim=None,
                 q_in=None, k_in=None, v_in=None, inner_proj=True):
        super().__init__()
        self.num_heads = num_heads
        inner_dim = dim if inner_dim is None else inner_dim
        self.inner_dim = inner_dim

        head_dim = inner_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, inner_dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(inner_dim, dim) if inner_proj else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv=None):
        if kv is None:
            kv = q
        B, N, C = kv.shape
        B_q, N_q, C_q = q.shape
        kv = self.kv(kv).reshape(B, N, 2, self.num_heads, self.inner_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q(q).reshape(B_q, N_q, self.num_heads, self.inner_dim // self.num_heads).permute(0, 2, 1, 3)  # B H N C

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_q, N_q, self.inner_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GRC_Self_Attention(nn.Module):
    def __init__(self, dim, attention_func=nn.MultiheadAttention, gr_cache=None, cache_ratio=0.5, decoder_attn=True,
                 spatial_pos_emb=True
                 , cls_dim=1, **kwargs):
        super().__init__()
        self.spatial_pos_emb = spatial_pos_emb
        self.cache_dim = int(dim * cache_ratio)
        self.decoder_attn = decoder_attn
        self.attn_self_func = attention_func(dim, **kwargs)
        self.attn_mem = attention_func(self.cache_dim, **kwargs)
        self.linear_reset = nn.Linear(2 * self.cache_dim, self.cache_dim)
        self.linear_update = nn.Linear(2 * self.cache_dim, self.cache_dim)
        self.linear_add = nn.Linear(2 * self.cache_dim, self.cache_dim)
        
        self.Norm1 = nn.LayerNorm(self.cache_dim)
        self.cls_dim = cls_dim
        self.heads = kwargs['num_heads']
        self.register_parameter('lam', nn.Parameter(torch.zeros([self.heads]) - 1))
        if self.spatial_pos_emb:
            self.pos_emb = nn.Conv2d(dim, dim, kernel_size=3, groups=self.cache_dim, stride=1, padding=1)

        self.memory_length = None
        if gr_cache is not None:
            self.gr_cache = gr_cache
        elif self.memory_length is not None:
            self.register_buffer('gr_cache', torch.zeros((1, self.memory_length, self.cache_dim,)))

    def forward(self, x,  **kwargs):
        B, T, C = x.shape

        if 'H' in kwargs:
            H, W = kwargs['H'], kwargs['W']
        else:
            H = W = int((T - self.cls_dim) ** 0.5)


        if not hasattr(self, 'gr_cache'):
            self.register_buffer('gr_cache', torch.zeros((1,) + (x.shape[1], self.cache_dim)).to(x.device))
            self.mH = H
            self.mW = W

        if self.spatial_pos_emb:
            B, T, C = x.shape
            spa_pos_emd = self.pos_emb(x[:, self.cls_dim:, :].view(B, H, W, C).permute(0, 3, 1, 2))
            spa_pos_emd = spa_pos_emd.view(B, -1, T - self.cls_dim).transpose(1, 2)
            spa_pos_emd = torch.cat([torch.zeros([B, self.cls_dim, C]).to(spa_pos_emd.device), spa_pos_emd],
                                    dim=1) if self.cls_dim > 0 else spa_pos_emd
        else:
            spa_pos_emd = 0

        x_self = self.attn_self_func(x, kv=x, **kwargs).view(B, T, self.heads, -1)

        gr_cache = self.gr_cache.to(x.device)
        gr_cache_value = gr_cache.expand((x.shape[0], gr_cache.shape[1], gr_cache.shape[-1]))

        x_summary = x_self.view_as(x)[:, :, :self.cache_dim]
        x_summary = f.interpolate(x_summary.transpose(1, 2), (self.gr_cache.shape[1])).transpose(1, 2)
        reset_gate = f.sigmoid(self.linear_reset(torch.cat([gr_cache_value, x_summary], dim=-1)))
        z_gate = f.sigmoid(self.linear_update(torch.cat([gr_cache_value, x_summary], dim=-1)))
        gr_cache_add = reset_gate * gr_cache_value
        gr_cache_add = self.Norm1(f.gelu(self.linear_add(torch.cat([gr_cache_add, x_summary], dim=-1))))
        gr_cache_value = z_gate * gr_cache_add + (1 - z_gate) * gr_cache_value

        if self.training:
            self.gr_cache.data = gr_cache_value.mean(dim=0, keepdims=True)

        x_mem = self.attn_mem(x[:, :, :self.cache_dim], gr_cache_value).view(B, T, self.heads, -1)
        alpha = self.lam.sigmoid().view(1, 1, -1, 1)

        return (alpha * torch.cat([x_mem, torch.zeros(B, T, self.heads, (C - self.cache_dim) // self.heads).to(x.device)],
                                  dim=-1) + (1 - alpha) * x_self).view_as(x) + spa_pos_emd
