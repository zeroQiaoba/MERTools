from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange, repeat

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # me: support window mask
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, mask=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), mask=mask))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# me: adapted from TimeSformer: https://github.com/facebookresearch/TimeSformer/blob/main/timesformer/models/vit.py
# factorised spatial temporal attention
class FSTBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, share_spatiotemporal_attn=False, temporal_first=False, temporal_seq_len=8):
        super().__init__()
        # spatial
        self.spatial_norm = norm_layer(dim)
        self.spatial_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # temporal
        self.share_spatiotemporal_attn = share_spatiotemporal_attn
        if share_spatiotemporal_attn:
            self.temporal_norm, self.temporal_attn = self.spatial_norm, self.spatial_attn
        else:
            self.temporal_norm = norm_layer(dim)
            self.temporal_attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn_norm = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_3 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2, self.gamma_3 = None, None, None

        self.temporal_first = temporal_first
        self.temporal_seq_len = temporal_seq_len # 8 when num_frames=16 and tuplet_size=2

    def forward(self, x):
        B, N, C = x.shape
        if self.gamma_1 is None:
            if self.temporal_first:
                x = rearrange(x, 'b (t hw) c -> (b hw) t c', t=self.temporal_seq_len, hw=N//self.temporal_seq_len)
                x = x + self.drop_path(self.temporal_attn(self.temporal_norm(x)))
                x = rearrange(x, '(b hw) t c -> (b t) hw c', t=self.temporal_seq_len, hw=N//self.temporal_seq_len)
                x = x + self.drop_path(self.spatial_attn(self.spatial_norm(x)))
                x = rearrange(x, '(b t) hw c -> b (t hw) c', t=self.temporal_seq_len, hw=N//self.temporal_seq_len)
                x = x + self.drop_path(self.mlp(self.ffn_norm(x)))
            else:
                x = rearrange(x, 'b (t hw) c -> (b t) hw c', t=self.temporal_seq_len, hw=N//self.temporal_seq_len)
                x = x + self.drop_path(self.spatial_attn(self.spatial_norm(x)))
                x = rearrange(x, '(b t) hw c -> (b hw) t c', t=self.temporal_seq_len, hw=N//self.temporal_seq_len)
                x = x + self.drop_path(self.temporal_attn(self.temporal_norm(x)))
                x = rearrange(x, '(b hw) t c -> b (t hw) c', t=self.temporal_seq_len, hw=N//self.temporal_seq_len)
                x = x + self.drop_path(self.mlp(self.ffn_norm(x)))
        else:
            if self.temporal_first:
                x = rearrange(x, 'b (t hw) c -> (b hw) t c', t=self.temporal_seq_len, hw=N//self.temporal_seq_len)
                x = x + self.drop_path(self.gamma_1 * self.temporal_attn(self.temporal_norm(x)))
                x = rearrange(x, '(b hw) t c -> (b t) hw c', t=self.temporal_seq_len, hw=N//self.temporal_seq_len)
                x = x + self.drop_path(self.gamma_2 * self.spatial_attn(self.spatial_norm(x)))
                x = rearrange(x, '(b t) hw c -> b (t hw) c', t=self.temporal_seq_len, hw=N//self.temporal_seq_len)
                x = x + self.drop_path(self.gamma_3 * self.mlp(self.ffn_norm(x)))
            else:
                x = rearrange(x, 'b (t hw) c -> (b t) hw c', t=self.temporal_seq_len, hw=N//self.temporal_seq_len)
                x = x + self.drop_path(self.gamma_1 * self.spatial_attn(self.spatial_norm(x)))
                x = rearrange(x, '(b t) hw c -> (b hw) t c', t=self.temporal_seq_len, hw=N//self.temporal_seq_len)
                x = x + self.drop_path(self.gamma_2 * self.temporal_attn(self.temporal_norm(x)))
                x = rearrange(x, '(b hw) t c -> b (t hw) c', t=self.temporal_seq_len, hw=N//self.temporal_seq_len)
                x = x + self.drop_path(self.gamma_3 * self.mlp(self.ffn_norm(x)))
        return x


"""
adapted from: https://github.com/google-research/scenic/blob/abacf900de5c1873b7c0cfcde616ac9eb22693bc/scenic/projects/token_learner/model.py#L113
"""
class TokenLearnerModuleV11(nn.Module):
    def __init__(self, dim, num_tokens, bottleneck_dim=64, drop=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 temporal_seq_len=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.norm = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=bottleneck_dim, out_features=num_tokens,
                       act_layer=act_layer, drop=drop)

        self.temporal_seq_len = temporal_seq_len

    def forward(self, x):
        attn = self.mlp(self.norm(x)) # (b, t*n1, n2), n2=num_tokens
        attn = torch.softmax(rearrange(attn, 'b (t n1) n2 -> (b t) n2 n1', t=self.temporal_seq_len), dim=-1)
        x = rearrange(x, 'b (t n1) c -> (b t) n1 c', t=self.temporal_seq_len)
        x = torch.matmul(attn, x)
        x = rearrange(x, '(b t) n2 c -> b (t n2) c', t=self.temporal_seq_len)
        return x

# me: move temporal_seq_len to forward
class TokenLearnerModuleV11_2(nn.Module):
    def __init__(self, dim, num_tokens, bottleneck_dim=64, drop=0., norm_layer=nn.LayerNorm, act_layer=nn.GELU):
        super().__init__()
        self.num_tokens = num_tokens
        self.norm = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=bottleneck_dim, out_features=num_tokens,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, temporal_seq_len):
        attn = self.mlp(self.norm(x)) # (b, t*n1, n2), n2=num_tokens
        attn = torch.softmax(rearrange(attn, 'b (t n1) n2 -> (b t) n2 n1', t=temporal_seq_len), dim=-1)
        x = rearrange(x, 'b (t n1) c -> (b t) n1 c', t=temporal_seq_len)
        x = torch.matmul(attn, x)
        x = rearrange(x, '(b t) n2 c -> b (t n2) c', t=temporal_seq_len)
        return x

class TokenFuser(nn.Module):
    def __init__(self, dim, num_tokens, bottleneck_dim=64, drop=0., temporal_seq_len=8, spatial_num_patches=196,
                 use_normalization=True, norm_layer=nn.LayerNorm, act_layer=nn.GELU,):
        super().__init__()

        self.norm1 = norm_layer(dim) if use_normalization else nn.Identity()
        self.norm2 = norm_layer(dim) if use_normalization else nn.Identity()

        self.temporal_seq_len = temporal_seq_len
        num_patches = num_tokens * num_tokens
        self.proj = nn.Linear(num_tokens, num_tokens) # Note: real implementation is different from the paper

        self.num_tokens = num_tokens
        self.mlp = Mlp(in_features=dim, hidden_features=bottleneck_dim, out_features=num_tokens,
                       act_layer=act_layer, drop=drop)

        self.norm3 = norm_layer(dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, bottleneck_x, x):
        bottleneck_x = rearrange(bottleneck_x, 'b (t n2) c -> (b t) n2 c', t=self.temporal_seq_len)
        x = rearrange(x, 'b (t n1) c -> (b t) n1 c', t=self.temporal_seq_len)

        bottleneck_x = rearrange(self.norm1(bottleneck_x), 'b n2 c -> b c n2') # b = b*t
        bottleneck_x = self.norm2(rearrange(self.proj(bottleneck_x), 'b c n2 -> b n2 c'))

        x = self.mlp(self.norm3(x)) # (b, n1, n2)
        x = torch.sigmoid(x)

        x = torch.matmul(x, bottleneck_x) # (b, n1, c)

        x = rearrange(self.dropout(x), '(b t) n1 c -> b (t n1) c', t=self.temporal_seq_len)

        return x


"""
adapted from https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
"""
# support cross attention
class GeneralAttention(nn.Module):
    def __init__(
            self, dim, context_dim=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.kv = nn.Linear(dim if context_dim is None else context_dim, all_head_dim * 2, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context=None):
        B, T1, C = x.shape
        q_bias, kv_bias = self.q_bias, None
        if self.q_bias is not None:
            kv_bias = torch.cat((torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, T1, self.num_heads, -1).transpose(1,2) # me: (B, H, T1, C//H)
        kv = F.linear(input=x if context is None else context, weight=self.kv.weight, bias=kv_bias)
        _, T2, _ = kv.shape
        kv = kv.reshape(B, T2, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # make torchscript happy (cannot use tensor as tuple), me： (B, H, T2, C//H)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # me: (B, H, T1, T2)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T1, -1) # (B, H, T1, C//H) -> (B, T1, H, C//H) -> (B, T1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PerceiverBlock(nn.Module):
    def __init__(self, num_layer, dim, context_dim,
                 num_cross_heads, num_heads,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_paths=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, cross_attn_head_dim=None):
        super().__init__()

        self.cross_attn = GeneralAttention(
            dim=dim, context_dim=context_dim, num_heads=num_cross_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=cross_attn_head_dim)
        self.cross_norm1 = norm_layer(dim)
        self.cross_norm2 = norm_layer(context_dim)

        self.cross_norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.cross_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        drop_paths = [drop_paths] * num_layer if not isinstance(drop_paths, list) else drop_paths
        self.self_attn_blocks =  nn.ModuleList([
            Block(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, drop_path=drop_paths[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(num_layer)
        ])

    def forward(self, x, context):
        latent = self.cross_attn(self.cross_norm1(x), context=self.cross_norm2(context))
        latent = self.cross_mlp(self.cross_norm3(latent))

        for self_attn_blk in self.self_attn_blocks:
            latent = self_attn_blk(latent)

        return latent


class Perceiver(nn.Module):
    def __init__(self, num_layer, num_latent_layer, dim, context_dim, num_latents,
                 num_cross_heads, num_heads,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_paths=None, init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, cross_attn_head_dim=None,
                 with_reverse_layer=True):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        assert len(drop_paths) == num_layer * num_latent_layer, \
            f"==> Error: len(drop_paths) ({len(drop_paths)}) != num_layer * num_latent_layer ({num_layer} * {num_latent_layer})."
        self.blks = nn.ModuleList([
            PerceiverBlock(
                num_layer=num_latent_layer, dim=dim, context_dim=context_dim,
                num_cross_heads=num_cross_heads, num_heads=num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                drop_paths=drop_paths[i*num_latent_layer:(i+1)*num_latent_layer], init_values=init_values, act_layer=act_layer, norm_layer=norm_layer,
                attn_head_dim=attn_head_dim, cross_attn_head_dim=cross_attn_head_dim)
            for i in range(num_layer)
        ])

        # MUST when pre-training
        if with_reverse_layer:
            self.reverse_blk = PerceiverBlock(
                    num_layer=0, dim=context_dim, context_dim=dim,
                    num_cross_heads=num_cross_heads, num_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                    drop_paths=0., init_values=init_values, act_layer=act_layer, norm_layer=norm_layer,
                    attn_head_dim=attn_head_dim, cross_attn_head_dim=cross_attn_head_dim)
        else: # optional when fine-tuning
            self.reverse_blk = None

    def forward(self, x):
        latent = repeat(self.latents, 'n d -> b n d', b=x.shape[0])

        for blk in self.blks:
            latent = blk(latent, context=x)

        x = self.reverse_blk(x, context=latent) if self.reverse_blk else latent

        return x


# cross + self
class CSBlock(nn.Module):
    def __init__(self, dim, context_dim, num_heads, num_cross_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, cross_attn_head_dim=None):
        super().__init__()

        self.cross_attn = GeneralAttention(
            dim=dim, context_dim=context_dim, num_heads=num_cross_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=cross_attn_head_dim)
        self.cross_norm1 = norm_layer(dim)
        self.cross_norm2 = norm_layer(context_dim)

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_0 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_0, self.gamma_1, self.gamma_2 = None, None, None

    def forward(self, x, context):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.cross_attn(self.cross_norm1(x), context=self.cross_norm2(context)))
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_0 * self.cross_attn(self.cross_norm1(x), context=self.cross_norm2(context)))
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# cross
class CrossBlock(nn.Module):
    def __init__(self, dim, context_dim, num_heads, num_cross_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, cross_attn_head_dim=None):
        super().__init__()

        self.cross_attn = GeneralAttention(
            dim=dim, context_dim=context_dim, num_heads=num_cross_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=cross_attn_head_dim)
        self.cross_norm1 = norm_layer(dim)
        self.cross_norm2 = norm_layer(context_dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, context):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.cross_attn(self.cross_norm1(x), context=self.cross_norm2(context)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.cross_attn(self.cross_norm1(x), context=self.cross_norm2(context)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# local + global
class LGBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None,
                 # new added
                 first_attn_type='self', third_attn_type='cross',
                 attn_param_sharing_first_third=False, attn_param_sharing_all=False,
                 no_second=False, no_third=False,
                 ):

        super().__init__()

        assert first_attn_type in ['self', 'cross'], f"Error: invalid attention type '{first_attn_type}', expected 'self' or 'cross'!"
        assert third_attn_type in ['self', 'cross'], f"Error: invalid attention type '{third_attn_type}', expected 'self' or 'cross'!"
        self.first_attn_type = first_attn_type
        self.third_attn_type = third_attn_type
        self.attn_param_sharing_first_third = attn_param_sharing_first_third
        self.attn_param_sharing_all = attn_param_sharing_all

        # Attention layer
        ## perform local (intra-region) attention, update messenger tokens
        ## (local->messenger) or (local<->local, local<->messenger)
        self.first_attn_norm0 = norm_layer(dim)
        if self.first_attn_type == 'cross':
            self.first_attn_norm1 = norm_layer(dim)
        self.first_attn = GeneralAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)

        ## perform global (inter-region) attention on messenger tokens
        ## (messenger<->messenger)
        self.no_second = no_second
        if not no_second:
            self.second_attn_norm0 = norm_layer(dim)
            if attn_param_sharing_all:
                self.second_attn = self.first_attn
            else:
                self.second_attn = GeneralAttention(
                    dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)

        ## perform local (intra-region) attention to inject global information into local tokens
        ## (messenger->local) or (local<->local, local<->messenger)
        self.no_third = no_third
        if not no_third:
            self.third_attn_norm0 = norm_layer(dim)
            if self.third_attn_type == 'cross':
                self.third_attn_norm1 = norm_layer(dim)
            if attn_param_sharing_first_third or attn_param_sharing_all:
                self.third_attn = self.first_attn
            else:
                self.third_attn = GeneralAttention(
                    dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)

        # FFN layer
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_0 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_0, self.gamma_1, self.gamma_2 = None, None, None

    # def forward(self, x):
    #     """
    #     :param x: (B, N, S, C),
    #         B: batch size
    #         N: number of local regions
    #         S: 1 + region size, 1: attached messenger token for each local region
    #         C: feature dim
    #     :return: (B, N, S, C),
    #     """
    #     b, n, s, c = x.shape
    #     if self.gamma_1 is None:
    #         # Attention layer
    #         ## perform local (intra-region) attention, update messenger tokens
    #         ## (local->messenger) or (local<->local, local<->messenger)
    #         if self.first_attn_type == 'self':
    #             x = rearrange(x, 'b n s c -> (b n) s c')  # s = 1+region_size
    #             x = x + self.drop_path(self.first_attn(self.first_attn_norm0(x)))
    #             x = rearrange(x, '(b n) s c -> b n s c', b=b)
    #         else: # 'cross'
    #             messenger_tokens = rearrange(x[:,:,:1], 'b n s c -> (b n) s c') # NOTE: ':1' for keeping dim, (B*N, 1, D)
    #             local_tokens = rearrange(x[:,:,1:], 'b n s c -> (b n) s c') # (B*N, S-1, D)
    #             messenger_tokens = messenger_tokens + self.drop_path(self.first_attn(self.first_attn_norm0(messenger_tokens), context=self.first_attn_norm1(local_tokens)))
    #             x[:,:,:1] = rearrange(messenger_tokens, '(b n) s c -> b n s c', b=b)
    #
    #         ## perform global (inter-region) attention on messenger tokens
    #         ## (messenger<->messenger)
    #         x[:,:,0] = x[:,:,0] + self.drop_path(self.second_attn(self.second_attn_norm0(x[:,:,0])))
    #
    #         ## perform local-global attention to inject global information into local tokens
    #         ## (messengers->local) or (local<->local, local<->messenger)
    #         if self.third_attn_type == 'self':
    #             x = rearrange(x, 'b n s c -> (b n) s c')  # s = 1+region_size
    #             x = x + self.drop_path(self.third_attn(self.third_attn_norm0(x)))
    #             x = rearrange(x, '(b n) s c -> b n s c', b=b)
    #         else:
    #             # 'cross', NOTE: all messengers acts as the source (key&value),
    #             # different from that (one messenger (as the target)) in the first attn.
    #             messenger_tokens = x[:,:,0] # NOTE: do not keep dim, (B, N, D)
    #             local_tokens = rearrange(x[:,:,1:], 'b n s c -> b (n s) c')# NOTE: n merges into s (not b), (B, N*(S-1), D)
    #             local_tokens = local_tokens + self.drop_path(self.third_attn(self.third_attn_norm0(local_tokens), context=self.third_attn_norm1(messenger_tokens)))
    #             x[:,:,1:] = rearrange(local_tokens, 'b (n s) c -> b n s c', n=n)
    #
    #         # FFN layer
    #         x = x + self.drop_path(self.mlp(self.norm2(x)))
    #     else:
    #         raise NotImplementedError
    #     return x

    def forward(self, x, b):
        """
        :param x: (B*N, S, C),
            B: batch size
            N: number of local regions
            S: 1 + region size, 1: attached messenger token for each local region
            C: feature dim
        param b: batch size
        :return: (B*N, S, C),
        """
        bn = x.shape[0]
        n = bn // b # number of local regions
        if self.gamma_1 is None:
            # Attention layer
            ## perform local (intra-region) attention, update messenger tokens
            ## (local->messenger) or (local<->local, local<->messenger)
            if self.first_attn_type == 'self':
                x = x + self.drop_path(self.first_attn(self.first_attn_norm0(x)))
            else: # 'cross'
                x[:,:1] = x[:,:1] + self.drop_path(
                    self.first_attn(
                        self.first_attn_norm0(x[:,:1]), # (b*n, 1, c)
                        context=self.first_attn_norm1(x[:,1:]) # (b*n, s-1, c)
                    )
                )

            ## perform global (inter-region) attention on messenger tokens
            ## (messenger<->messenger)
            if not self.no_second:
                messenger_tokens = rearrange(x[:,0], '(b n) c -> b n c', b=b) # attn on 'n' dim
                messenger_tokens = messenger_tokens + self.drop_path(
                    self.second_attn(self.second_attn_norm0(messenger_tokens))
                )
                x[:,0] = rearrange(messenger_tokens, 'b n c -> (b n) c')
            else: # for usage in the third attn
                messenger_tokens = rearrange(x[:,0], '(b n) c -> b n c', b=b) # attn on 'n' dim

            ## perform local-global attention to inject global information into local tokens
            ## (messengers->local) or (local<->local, local<->messenger)
            if not self.no_third:
                if self.third_attn_type == 'self':
                    x = x + self.drop_path(self.third_attn(self.third_attn_norm0(x)))
                else:
                    # 'cross', NOTE: all messengers acts as the source (key&value),
                    # different from that (one messenger (as the target)) in the first attn.
                    local_tokens = rearrange(x[:,1:], '(b n) s c -> b (n s) c', b=b)# NOTE: n merges into s (not b), (B, N*(S-1), D)
                    local_tokens = local_tokens + self.drop_path(
                        self.third_attn(
                            self.third_attn_norm0(local_tokens), # (b, n*(s-1), c)
                            context=self.third_attn_norm1(messenger_tokens) # (b, n*1, c)
                        )
                    )
                    x[:,1:] = rearrange(local_tokens, 'b (n s) c -> (b n) s c', n=n)

            # FFN layer
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            raise NotImplementedError
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # me: for more attention types
        self.temporal_seq_len = num_frames // self.tubelet_size
        self.spatial_num_patches = num_patches // self.temporal_seq_len
        self.input_token_size = (num_frames // self.tubelet_size, img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False, 
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_mean_pooling=True,
                 keep_temporal_dim=False, # do not perform temporal pooling, has higher priority than 'use_mean_pooling'
                 head_activation_func=None, # activation function after head fc, mainly for the regression task
                 attn_type='joint',
                 fst_share_st_attn=False, fst_temporal_first=False, # for factorised attention
                 tf_start_layer=7, tf_num_tokens=8, tf_bottleneck_dim=64,  # for token fuser
                 p_start_layer=7, p_num_latents=16, p_num_layer=3, p_num_latent_layer=2,  # for perceiver
                 p_dim=768, p_num_cross_heads=4,  # for perceiver
                 part_win_size=(2,2,10), part_cls_type='org', part_local_first=False, # for part window attention
                 tem_win_size=(1,2,4,8), tem_win_depth=(6,6,6,6), # for temporal window attention
                 tem_pyr_depth=(8,8,8), tem_pyr_kernel_size=2, tem_pyr_stride=2, tem_pyr_type='avg', # for temporal pyramid
                 tem_pyr_no_use_multiscale_feature=False, tem_pyr_type_up='repeat',
                 st_pyr_depth=(12, 8, 4), st_pyr_kernel_size=2, st_pyr_stride=2, st_pyr_type='conv', # for spatial temporal pyramid
                 st_pyr_type_up='repeat', st_tf_num_tokens=(8,), st_tf_bottleneck_dim=128, st_pyr_no_multiscale_feature=False,
                 st_pyr_spatial_only_cross_attn=False, st_pyr_window_size=None,
                 st_pyr_disable_temporal_pyr=False, st_pyr_disable_spatial_pyr=False,
                 lg_region_size=(2, 2, 10), lg_first_attn_type='self', lg_third_attn_type='cross',  # for local_global
                 lg_attn_param_sharing_first_third=False, lg_attn_param_sharing_all=False,
                 lg_classify_token_type='org', lg_no_second=False, lg_no_third=False,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # me: support more attention types
        self.attn_type = attn_type
        if attn_type == 'joint' or attn_type == 'only_spatial':
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values)
                for i in range(depth)])
        elif attn_type == 'factorised':
            print(f"==> Note: Use factorised spatiotemporal attention (share_spatiotemporal_attn={fst_share_st_attn}, temporal_first={fst_temporal_first})")
            self.blocks = nn.ModuleList([
                FSTBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values,
                    share_spatiotemporal_attn=fst_share_st_attn, temporal_first=fst_temporal_first,
                    temporal_seq_len=self.patch_embed.temporal_seq_len
                )
                for i in range(depth)])
        elif attn_type == 'factorised2': # spatial attn + ffn -> temporal attn + ffn
            print(f"==> Note: Use factorised2 spatiotemporal attention (temporal_first={fst_temporal_first})")
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values)
                for i in range(depth)])
            self.temporal_first = fst_temporal_first
        elif attn_type == 'tokenfuser':
            print(f"==> Note: Use token fuser for compute reduction (start_layer={tf_start_layer}, num_tokens={tf_num_tokens}, bottleneck_dim={tf_bottleneck_dim})")
            self.blocks = nn.ModuleList([])
            self.tf_start_layer = tf_start_layer
            for i in range(depth):
                if (i+1) < tf_start_layer: # token fuser start layer
                    self.blocks.append(
                        Block(
                            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                            init_values=init_values)
                    )
                else:
                    block = nn.ModuleList([])
                    block.append(
                        TokenLearnerModuleV11(
                            dim=embed_dim, num_tokens=tf_num_tokens, bottleneck_dim=tf_bottleneck_dim,
                            drop=drop_rate, norm_layer=norm_layer, temporal_seq_len=self.patch_embed.temporal_seq_len
                        )
                    )
                    block.append(
                        Block(
                            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                            init_values=init_values)
                    )
                    block.append(
                        TokenFuser(
                            dim=embed_dim, num_tokens=tf_num_tokens, bottleneck_dim=tf_bottleneck_dim,
                            drop=drop_rate, norm_layer=norm_layer, temporal_seq_len=self.patch_embed.temporal_seq_len,
                            spatial_num_patches=self.patch_embed.spatial_num_patches
                        )
                    )
                    self.blocks.append(block)
        elif attn_type == 'perceiver':
            print(f"==> Note: Use perceiver for compute reduction (start_layer={p_start_layer}, num_latents={p_num_latents}, "
                  f"num_layer={p_num_layer}, num_latent_layer={p_num_latent_layer}, "
                  f"dim={p_dim}, num_cross_heads={p_num_cross_heads})")
            self.blocks = nn.ModuleList([])
            num_normal_blks = depth - p_start_layer + 1
            for i in range(num_normal_blks):
                self.blocks.append(
                    Block(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                        init_values=init_values)
                )
            self.blocks.append(
                Perceiver(
                    num_layer=p_num_layer, num_latent_layer=p_num_latent_layer,
                    dim=p_dim, context_dim=embed_dim, num_latents=p_num_latents,
                    num_cross_heads=p_num_cross_heads, num_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_paths=dpr[(p_start_layer-1):], init_values=init_values, norm_layer=norm_layer,
                    with_reverse_layer=True)
            )
        elif  attn_type == 'part_window':
            print(f"==> Note: Use part_window for compute reduction (part_win_size={part_win_size}, part_cls_type={part_cls_type}, part_local_first={part_local_first})")
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values)
                for i in range(depth)])
            # part tokens
            self.part_win_size = part_win_size # (t, h, w)
            self.part_size = list(i//j for i,j in zip(self.patch_embed.input_token_size, part_win_size)) # (nt, nh, nw)
            num_spatial_parts = self.part_size[1] * self.part_size[2] # nh * nw
            self.part_tokens = nn.Parameter(torch.zeros(num_spatial_parts, embed_dim)) # (1, num_parts, 1, C)
            trunc_normal_(self.part_tokens, std=.02)
            self.part_local_first = part_local_first
            # cls
            self.part_cls_type = part_cls_type
        elif attn_type == 'temporal_window':
            print(f"==> Note: Use temporal_window for compute reduction (tem_win_size={tem_win_size}, tem_win_depth={tem_win_depth})")
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values)
                for i in range(depth)])
            assert sum(tem_win_depth) == depth, 'Error: the sum of temporal window depth != total depth!'
            assert len(tem_win_size) == len(tem_win_depth)
            self.tem_win_depth = tem_win_depth
            self.tem_win_size =  tem_win_size
        # elif attn_type == 'temporal_pyramid':
        #     print(f"==> Note: Use temporal_pyramid for compute reduction (tem_pyr_type={tem_pyr_depth}, tem_pyr_depth={tem_pyr_depth}, "
        #           f"tem_pyr_kernel_size={tem_pyr_kernel_size}, tem_pyr_stride={tem_pyr_stride})")
        #     self.blocks = nn.ModuleList([
        #         Block(
        #             dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
        #             init_values=init_values)
        #         for i in range(depth)])
        #     assert sum(tem_pyr_depth) == depth, 'Error: the sum of temporal window depth != total depth!'
        #     self.tem_pyr_depth = tem_pyr_depth
        #     self.tem_pyr_type = tem_pyr_type
        #     self.tem_pyr_kernel_size = tem_pyr_kernel_size
        #     self.tem_pyr_stride = tem_pyr_stride
        #     self.tem_pyr_use_multiscale_feature = tem_pyr_use_multiscale_feature
        #     # downsample modules
        #     self.downsamples = nn.ModuleList([])
        #     for _ in range(len(tem_pyr_depth) - 1):
        #         if tem_pyr_type == 'avg':
        #             self.downsamples.append(nn.AvgPool1d(kernel_size=tem_pyr_kernel_size, stride=tem_pyr_stride))
        #         elif tem_pyr_type == 'max':
        #             self.downsamples.append(nn.MaxPool1d(kernel_size=tem_pyr_kernel_size, stride=tem_pyr_stride))
        #         elif tem_pyr_type == 'conv':
        #             self.downsamples.append(nn.Conv1d(embed_dim, embed_dim, kernel_size=tem_pyr_kernel_size, stride=tem_pyr_stride))
        #         else:
        #             raise NotImplementedError

        elif attn_type == 'temporal_pyramid':
            print(f"==> Note: Use temporal_pyramid for compute reduction (tem_pyr_type={tem_pyr_depth}, tem_pyr_type_up={tem_pyr_type_up}"
                  f"tem_pyr_depth={tem_pyr_depth}, tem_pyr_kernel_size={tem_pyr_kernel_size}, tem_pyr_stride={tem_pyr_stride},"
                  f"tem_pyr_no_use_multiscale_feature={tem_pyr_no_use_multiscale_feature})")
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values)
                for i in range(depth)])
            assert sum(tem_pyr_depth) == depth, 'Error: the sum of temporal window depth != total depth!'
            self.tem_pyr_depth = tem_pyr_depth
            self.tem_pyr_type = tem_pyr_type # downsample
            self.tem_pyr_kernel_size = tem_pyr_kernel_size
            self.tem_pyr_stride = tem_pyr_stride
            self.tem_pyr_type_up = tem_pyr_type_up # upsample

            self.tem_pyr_no_use_multiscale_feature = tem_pyr_no_use_multiscale_feature

            # downsample modules
            self.downsamples = nn.ModuleList([])
            self.upsamples = nn.ModuleList([])
            for pyr_idx in range(1, len(tem_pyr_depth)):
                if tem_pyr_type == 'avg':
                    self.downsamples.append(nn.AvgPool1d(kernel_size=tem_pyr_kernel_size, stride=tem_pyr_stride))
                elif tem_pyr_type == 'max':
                    self.downsamples.append(nn.MaxPool1d(kernel_size=tem_pyr_kernel_size, stride=tem_pyr_stride))
                elif tem_pyr_type == 'conv':
                    self.downsamples.append(nn.Conv1d(embed_dim, embed_dim, kernel_size=tem_pyr_kernel_size, stride=tem_pyr_stride))
                else:
                    raise NotImplementedError
                if tem_pyr_type_up == 'conv':
                    self.upsamples.append(nn.ConvTranspose1d(embed_dim, embed_dim,
                                                             kernel_size=tem_pyr_kernel_size ** pyr_idx,
                                                             stride=tem_pyr_stride ** pyr_idx))

        elif attn_type == 'st_pyramid': # spatial temporal token reduction
            print(f"==> Note: Use st_pyramid for compute reduction (st_pyr_type={st_pyr_type}, st_pyr_type_up={st_pyr_type_up}, "
                  f"st_pyr_depth={st_pyr_depth}, st_pyr_kernel_size={st_pyr_kernel_size}, st_pyr_stride={st_pyr_stride}, "
                  f"st_tf_num_tokens={st_tf_num_tokens}, st_tf_bottleneck_dim={st_tf_bottleneck_dim}, "
                  f"st_pyr_no_multiscale_feature={st_pyr_no_multiscale_feature}, "
                  f"st_pyr_spatial_only_cross_attn={st_pyr_spatial_only_cross_attn},"
                  f"st_pyr_window_size={st_pyr_window_size}, "
                  f"st_pyr_disable_temporal_pyr={st_pyr_disable_temporal_pyr}, "
                  f"st_pyr_disable_spatial_pyr={st_pyr_disable_spatial_pyr})")

            # me: re-assign model depth!!!
            print(f"==> Note: re-assign model depth from default {depth} to {sum(st_pyr_depth)} (sum({st_pyr_depth}))")
            depth = sum(st_pyr_depth)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
            # assert sum(st_pyr_depth) == depth, 'Error: the sum of temporal window depth != total depth!'
            self.st_pyr_depth = st_pyr_depth
            self.st_pyr_type = st_pyr_type # temporal downsample
            self.st_pyr_kernel_size = st_pyr_kernel_size
            self.st_pyr_stride = st_pyr_stride
            self.st_pyr_type_up = st_pyr_type_up # temporal upsample

            # self.st_tf_num_tokens = st_tf_num_tokens # spatial downsample
            
            # support multi tokens
            if len(st_tf_num_tokens) == 1:
                self.st_tf_num_tokens = st_tf_num_tokens * (len(st_pyr_depth) - 1) # spatial downsample
            else:
                self.st_tf_num_tokens = st_tf_num_tokens
            self.st_tf_bottleneck_dim = st_tf_bottleneck_dim

            self.st_pyr_spatial_only_cross_attn = st_pyr_spatial_only_cross_attn

            self.st_pyr_no_multiscale_feature = st_pyr_no_multiscale_feature

            # for applying 3d window based attention (instead of free global attention) in the first stage
            self.st_pyr_window_size = st_pyr_window_size
            if st_pyr_window_size is not None:
                assert st_pyr_depth[0] % 2 == 0, f'Error: first stage length is expected to be even, got {st_pyr_depth[0]}!'
                self.st_pyr_shift_size = tuple(i // 2 for i in st_pyr_window_size)
                print(f"==> Note: use shifted window attention (window_size={self.st_pyr_window_size}, shift_size={self.st_pyr_shift_size}) in the first stage!")
            else:
                self.st_pyr_shift_size = None

            self.blocks = nn.ModuleList([])
            self.temporal_downsamples = nn.ModuleList([])
            self.temporal_upsamples = nn.ModuleList([])
            self.spatial_downsamples = nn.ModuleList([])
            self.spatial_upsamples = nn.ModuleList([])

            self.st_pyr_disable_temporal_pyr = st_pyr_disable_temporal_pyr # no temporal pyramid
            self.st_pyr_disable_spatial_pyr = st_pyr_disable_spatial_pyr # no spatial pyramid

            block_idx = 0
            for pyr_idx, depth in enumerate(st_pyr_depth):
                if pyr_idx == 0:
                    for _ in range(depth):
                        self.blocks.append(
                            Block(
                                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], norm_layer=norm_layer,
                                init_values=init_values)
                        )
                        block_idx += 1
                else:
                    # temporal downsample/upsample
                    if not self.st_pyr_disable_temporal_pyr:
                        if st_pyr_type == 'avg':
                            self.temporal_downsamples.append(nn.AvgPool1d(kernel_size=st_pyr_kernel_size, stride=st_pyr_stride))
                        elif st_pyr_type == 'max':
                            self.temporal_downsamples.append(nn.MaxPool1d(kernel_size=st_pyr_kernel_size, stride=st_pyr_stride))
                        elif st_pyr_type == 'conv':
                            self.temporal_downsamples.append(nn.Conv1d(embed_dim, embed_dim, kernel_size=st_pyr_kernel_size, stride=st_pyr_stride))
                        else:
                            raise NotImplementedError
                        if st_pyr_type_up == 'conv':
                            self.temporal_upsamples.append(nn.ConvTranspose1d(embed_dim, embed_dim,
                                                                              kernel_size=st_pyr_kernel_size ** pyr_idx,
                                                                              stride=st_pyr_stride ** pyr_idx))
                    else:
                        print(f'==> Note: disable temporal pyramid, no temporal downsample and upsample!')
                    # spatial downsample
                    if not self.st_pyr_disable_spatial_pyr:
                        cur_temporal_seq_len = self.patch_embed.temporal_seq_len // (st_pyr_stride ** pyr_idx)
                        self.spatial_downsamples.append(
                            TokenLearnerModuleV11(
                                dim=embed_dim, num_tokens=self.st_tf_num_tokens[pyr_idx-1], bottleneck_dim=st_tf_bottleneck_dim,
                                drop=drop_rate, norm_layer=norm_layer, temporal_seq_len=cur_temporal_seq_len
                            )
                        )
                        # cross self blocks, the last one is for spatial upsample
                        if self.st_pyr_spatial_only_cross_attn:
                            block_class = CrossBlock
                            print("==> Note: use 'CrossBlock' instead of 'CSBlock' !")
                        else:
                            block_class = CSBlock
                        for _ in range(depth):
                            self.blocks.append(
                                block_class(dim=embed_dim, context_dim=embed_dim, num_heads=num_heads, num_cross_heads=num_heads,
                                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[block_idx], norm_layer=norm_layer, init_values=init_values)
                            )
                            block_idx += 1
                    else:
                        print(f'==> Note: disable spatial pyramid, use normal global attention block instead!')
                        for _ in range(depth):
                            self.blocks.append(
                                Block(
                                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], norm_layer=norm_layer,
                                    init_values=init_values)
                            )
                            block_idx += 1
        elif attn_type == 'tokenfuser2':
            print(f"==> Note: Use token fuser (version 2) for compute reduction (start_layer={tf_start_layer}, num_tokens={tf_num_tokens}, bottleneck_dim={tf_bottleneck_dim})")
            self.blocks = nn.ModuleList([])
            self.tf_start_layer = tf_start_layer
            for i in range(depth):
                self.blocks.append(
                    Block(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                        init_values=init_values)
                )
            self.spatial_downsample = TokenLearnerModuleV11(
                dim=embed_dim, num_tokens=tf_num_tokens, bottleneck_dim=tf_bottleneck_dim,
                drop=drop_rate, norm_layer=norm_layer, temporal_seq_len=self.patch_embed.temporal_seq_len
            )
            # self.spatial_upsample = TokenFuser(
            #     dim=embed_dim, num_tokens=tf_num_tokens, bottleneck_dim=tf_bottleneck_dim,
            #     drop=drop_rate, norm_layer=norm_layer, temporal_seq_len=self.patch_embed.temporal_seq_len,
            #     spatial_num_patches=self.patch_embed.spatial_num_patches
            # )
        elif attn_type == 'local_global':
            print(f"==> Note: Use 'local_global' for compute reduction (lg_region_size={lg_region_size},"
                  f"lg_first_attn_type={lg_first_attn_type}, lg_third_attn_type={lg_third_attn_type},"
                  f"lg_attn_param_sharing_first_third={lg_attn_param_sharing_first_third},"
                  f"lg_attn_param_sharing_all={lg_attn_param_sharing_all},"
                  f"lg_classify_token_type={lg_classify_token_type},"
                  f"lg_no_second={lg_no_second}, lg_no_third={lg_no_third})")
            self.blocks = nn.ModuleList([
                LGBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values,
                    first_attn_type=lg_first_attn_type, third_attn_type=lg_third_attn_type,
                    attn_param_sharing_first_third=lg_attn_param_sharing_first_third,
                    attn_param_sharing_all=lg_attn_param_sharing_all,
                    no_second=lg_no_second, no_third=lg_no_third,
                )
                for i in range(depth)])
            # region tokens
            self.lg_region_size = lg_region_size # (t, h, w)
            self.lg_num_region_size = list(i//j for i,j in zip(self.patch_embed.input_token_size, lg_region_size)) # (nt, nh, nw)
            num_regions = self.lg_num_region_size[0] * self.lg_num_region_size[1] * self.lg_num_region_size[2] # nt * nh * nw
            print(f"==> Number of local regions: {num_regions} (size={self.lg_num_region_size})")
            self.lg_region_tokens = nn.Parameter(torch.zeros(num_regions, embed_dim))
            trunc_normal_(self.lg_region_tokens, std=.02)

            # The token type used for final classification
            self.lg_classify_token_type = lg_classify_token_type
            assert lg_classify_token_type in ['org', 'region', 'all'], \
                f"Error: wrong 'lg_classify_token_type' in local_global attention ('{lg_classify_token_type}'), expected 'org'/'region'/'all'!"

        else:
            raise NotImplementedError

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # me: add frame-level prediction support
        self.keep_temporal_dim = keep_temporal_dim

        # me: add head activation function support for regression task
        if head_activation_func is not None:
            if head_activation_func == 'sigmoid':
                self.head_activation_func = nn.Sigmoid()
            elif head_activation_func == 'relu':
                self.head_activation_func = nn.ReLU()
            elif head_activation_func == 'tanh':
                self.head_activation_func = nn.Tanh()
            else:
                raise NotImplementedError
        else: # default
            self.head_activation_func = nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'part_tokens', 'lg_region_tokens'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, _, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        if self.attn_type == 'tokenfuser':
            for i, blk in enumerate(self.blocks, 1):
                if i < self.tf_start_layer:
                    x = blk(x)
                else:
                    residual = x
                    tf_learner, tf_encoder, tf_fuser = blk
                    x = tf_learner(x)
                    x = tf_encoder(x)
                    x = tf_fuser(x, residual)
                    x = residual + x
        elif self.attn_type == 'only_spatial':
            x = rearrange(x, 'b (t hw) c -> (b t) hw c', t=self.patch_embed.temporal_seq_len)
            for blk in self.blocks:
                x = blk(x)
            x = rearrange(x, '(b t) hw c -> b (t hw) c', t=self.patch_embed.temporal_seq_len)
        elif self.attn_type == 'part_window':
            # input: window partition
            nt, t = self.part_size[0], self.part_win_size[0]
            nh, h = self.part_size[1], self.part_win_size[1]
            nw, w = self.part_size[2], self.part_win_size[2]
            b = x.size(0)
            x = rearrange(x, 'b (nt t nh h nw w) c -> b (nt nh nw) (t h w) c', nt=nt,t=t,nh=nh,h=h,nw=nw,w=w)
            # add part tokens
            part_tokens = repeat(self.part_tokens, 'n c -> b (nt n) 1 c', b=b, nt=nt)
            x = torch.cat([part_tokens, x], dim=2) # (b, nt*nh*nw, 1+thw, c)
            mod = 0 if not self.part_local_first else 1
            for i, blk in enumerate(self.blocks):
                # if i % 2 == 0 and not self.part_second: # part token attn for spatiotemporal modeling
                if i % 2 == mod: # part token attn for spatiotemporal modeling
                    x[:,:,0] = blk(x[:,:,0])
                else: # part attn for spatial modeling
                    x = rearrange(x, 'b n s c -> (b n) s c') # s = thw
                    x = blk(x)
                    x = rearrange(x, '(b n) s c -> b n s c', b=b)

            if self.part_cls_type == 'part': # only use part tokens for classification
                x = x[:,:,0] # (b, n, c)
            elif self.part_cls_type == 'org': # only use original tokens for classification
                x = rearrange(x[:,:,1:], 'b n s c -> b (n s) c') # s = thw
            else: # use all tokens for classification
                x = rearrange(x, 'b n s c -> b (n s) c') # s = 1 + thw
        elif self.attn_type == 'temporal_window':
            i = 0
            for size, depth in zip(self.tem_win_size, self.tem_win_depth):
                x = rearrange(x, 'b (nt t) c -> (b nt) t c', nt=self.patch_embed.temporal_seq_len//size)
                for _ in range(depth):
                    x = self.blocks[i](x)
                    i += 1
                x = rearrange(x, '(b nt) t c -> b (nt t) c', nt=self.patch_embed.temporal_seq_len//size)
        elif self.attn_type == 'factorised2': # spatial attn + ffn -> temporal attn + ffn
            mod = 0 if not self.temporal_first else 1
            b = x.shape[0]
            for i, blk in enumerate(self.blocks):
                if i % 2 == mod: # spatial attn
                    x = rearrange(x, 'b (t hw) c -> (b t) hw c', t=self.patch_embed.temporal_seq_len)
                    x = blk(x)
                    x = rearrange(x, '(b t) hw c -> b (t hw) c', t=self.patch_embed.temporal_seq_len)
                else: # temporal attn
                    x = rearrange(x, 'b (t hw) c -> (b hw) t c', t=self.patch_embed.temporal_seq_len)
                    x = blk(x)
                    x = rearrange(x, '(b hw) t c -> b (t hw) c', b=b)
                    
        # elif self.attn_type == 'temporal_pyramid':
        #     i = 0
        #     b = x.shape[0]
        #     multiscale_features = []
        #     for pyr_idx, depth in enumerate(self.tem_pyr_depth):
        #         for _ in range(depth):
        #             x = self.blocks[i](x)
        #             i += 1
        #         multiscale_features.append(x)
        #         # downsample
        #         if pyr_idx < (len(self.tem_pyr_depth) - 1):
        #             # me: bug for t because t is not fixed
        #             # x = rearrange(x, 'b (t hw) c -> (b hw) c t', t=self.patch_embed.temporal_seq_len)
        #             x = rearrange(x, 'b (t hw) c -> (b hw) c t', hw=self.patch_embed.spatial_num_patches)
        #             x = self.downsamples[pyr_idx](x)
        #             x = rearrange(x, '(b hw) c t -> b (t hw) c', b=b)
        #     if self.tem_pyr_use_multiscale_feature:
        #         x = torch.stack([feat.mean(1) for feat in multiscale_features], dim=1)

        elif self.attn_type == 'temporal_pyramid':
            i = 0
            b = x.shape[0]
            t = self.patch_embed.temporal_seq_len
            hw = x.shape[1] // t
            multiscale_feature = None
            for pyr_idx, depth in enumerate(self.tem_pyr_depth):
                for _ in range(depth):
                    x = self.blocks[i](x)
                    i += 1
                # upsample
                if pyr_idx == 0:
                    multiscale_feature = x
                else:
                    if self.tem_pyr_type_up == 'conv':
                        x_up = rearrange(x, 'b (t hw) c -> (b hw) c t', hw=hw)
                        x_up = self.upsamples[pyr_idx - 1](x_up)
                        x_up = rearrange(x_up, '(b hw) c t -> b (t hw) c', b=b)
                    else:
                        x_up = rearrange(x, 'b (t hw) c -> b t hw c', hw=hw)
                        x_up = x_up.repeat_interleave(self.tem_pyr_stride ** pyr_idx, dim=1)
                        x_up = rearrange(x_up, 'b t hw c -> b (t hw) c')
                    multiscale_feature = multiscale_feature + x_up
                # downsample
                if pyr_idx < (len(self.tem_pyr_depth) - 1):
                    x = rearrange(x, 'b (t hw) c -> (b hw) c t', hw=hw)
                    x = self.downsamples[pyr_idx](x)
                    x = rearrange(x, '(b hw) c t -> b (t hw) c', b=b)
            # re-assign
            if not self.tem_pyr_no_use_multiscale_feature:
                x = multiscale_feature

        elif self.attn_type == 'st_pyramid': # spatial temporal token reduction
            block_idx = 0
            b = x.shape[0] # x.shape = 'b (t h w) c'
            multiscale_feature = None
            for pyr_idx, depth in enumerate(self.st_pyr_depth):
                if pyr_idx == 0:
                    if self.st_pyr_window_size is None:
                        for _ in range(depth):
                            x = self.blocks[block_idx](x)
                            block_idx += 1
                    else:
                        # reshape
                        # t, h, w = self.patch_embed.input_token_size
                        # x = rearrange(x, 'b (t h w) c -> b c t h w', t=t, h=h, w=w)

                        # calculate attention mask for SW-MSA
                        D, H, W = self.patch_embed.input_token_size
                        B, _, C = x.shape
                        # B, C, D, H, W = x.shape
                        window_size, shift_size = get_window_size((D, H, W), self.st_pyr_window_size, self.st_pyr_shift_size)
                        x = rearrange(x, 'b (d h w) c -> b d h w c', d=D, h=H, w=W)
                        # x = rearrange(x, 'b c d h w -> b d h w c')

                        # me: for no padding
                        # assert D % window_size[0] == 0 and H % window_size[1] == 0 and W % window_size[2] == 0

                        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
                        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
                        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
                        mask_matrix = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
                        for _ in range(depth):
                            # set shift size
                            if block_idx % 2 == 0:
                                shift_size = (0, 0, 0)
                            else:
                                shift_size = self.st_pyr_shift_size

                            # adapted from SwinTransformerBlock3D in video swin transformer
                            window_size, shift_size = get_window_size((D, H, W), window_size, shift_size)

                            # pad feature maps to multiples of window size
                            pad_l = pad_t = pad_d0 = 0
                            pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
                            pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
                            pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
                            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
                            _, Dp, Hp, Wp, _ = x.shape
                            # cyclic shift
                            if any(i > 0 for i in shift_size):
                                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
                                attn_mask = mask_matrix
                            else:
                                shifted_x = x
                                attn_mask = None
                            # partition windows
                            x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
                            # W-MSA/SW-MSA + MLP
                            attn_windows = self.blocks[block_idx](x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
                            # merge windows
                            attn_windows = attn_windows.view(-1, *(window_size + (C,)))
                            shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
                            # reverse cyclic shift
                            if any(i > 0 for i in shift_size):
                                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
                            else:
                                x = shifted_x

                            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                                x = x[:, :D, :H, :W, :].contiguous()

                            # next block
                            block_idx += 1

                        # reshape back
                        x = rearrange(x, 'b d h w c -> b (d h w) c')

                    # store
                    multiscale_feature = x
                else:
                    # temporal downsample
                    if not self.st_pyr_disable_temporal_pyr:
                        cur_temporal_seq_len = self.patch_embed.temporal_seq_len // (self.st_pyr_stride ** (pyr_idx - 1) )
                        x = rearrange(x, 'b (t hw) c -> (b hw) c t', t=cur_temporal_seq_len)
                        x = self.temporal_downsamples[pyr_idx - 1](x)
                        x = rearrange(x, '(b hw) c t -> b (t hw) c', b=b)
                        cur_temporal_seq_len = cur_temporal_seq_len // self.st_pyr_stride
                    if not self.st_pyr_disable_spatial_pyr:
                        x_before_spatial_downsample = x
                        # spatial downsample
                        x = self.spatial_downsamples[pyr_idx - 1](x)
                        # cross self block
                        for _ in range(depth - 1):
                            x = self.blocks[block_idx](x, context=x_before_spatial_downsample)
                            block_idx += 1
                        # spatial upsample
                        x = self.blocks[block_idx](x_before_spatial_downsample, context=x)
                        block_idx += 1
                    else:
                        # normal block
                        for _ in range(depth):
                            x = self.blocks[block_idx](x)
                            block_idx += 1
                    # temporal upsample
                    if not self.st_pyr_disable_temporal_pyr:
                        if self.st_pyr_type_up == 'conv':
                            x_up = rearrange(x, 'b (t hw) c -> (b hw) c t', t=cur_temporal_seq_len)
                            x_up = self.temporal_upsamples[pyr_idx - 1](x_up)
                            x_up = rearrange(x_up, '(b hw) c t -> b (t hw) c', b=b)
                        else:
                            x_up = rearrange(x, 'b (t hw) c -> b t hw c', t=cur_temporal_seq_len)
                            x_up = x_up.repeat_interleave(self.st_pyr_stride ** pyr_idx, dim=1)
                            x_up = rearrange(x_up, 'b t hw c -> b (t hw) c')
                        multiscale_feature = multiscale_feature + x_up
                    else:
                        # if no temporal pyramid, do not use multiscale feature
                        multiscale_feature = x
            # re-assign
            if not self.st_pyr_no_multiscale_feature:
                x = multiscale_feature

        elif self.attn_type == 'local_global':
            # input: region partition
            nt, t = self.lg_num_region_size[0], self.lg_region_size[0]
            nh, h = self.lg_num_region_size[1], self.lg_region_size[1]
            nw, w = self.lg_num_region_size[2], self.lg_region_size[2]
            # NOTE: during fine-tuning, use seperate h and w, different from that (hw) in pre-training
            # nhw = self.lg_num_region_size[1] * self.lg_num_region_size[2]
            # hw = int(self.lg_region_size[1] * self.lg_region_size[2])
            b = x.size(0)
            # x = rearrange(x, 'b (nt t nhw hw) c -> b (nt nhw) (t hw) c', nt=nt,t=t,nhw=nhw,hw=hw)
            x = rearrange(x, 'b (nt t nh h nw w) c -> b (nt nh nw) (t h w) c', nt=nt,nh=nh,nw=nw,t=t,h=h,w=w)
            # add region tokens
            region_tokens = repeat(self.lg_region_tokens, 'n c -> b n 1 c', b=b)
            x = torch.cat([region_tokens, x], dim=2) # (b, nt*nh*nw, 1+thw, c)
            x = rearrange(x, 'b n s c -> (b n) s c') # s = 1 + thw
            # run through each block
            for blk in self.blocks:
                x = blk(x, b) # (b*n, s, c)

            x = rearrange(x, '(b n) s c -> b n s c', b=b) # s = 1 + thw
            # token for final classification
            if self.lg_classify_token_type == 'region': # only use region tokens for classification
                x = x[:,:,0] # (b, n, c)
            elif self.lg_classify_token_type == 'org': # only use original tokens for classification
                x = rearrange(x[:,:,1:], 'b n s c -> b (n s) c') # s = thw
            else: # use all tokens for classification
                x = rearrange(x, 'b n s c -> b (n s) c') # s = 1 + thw

        else:
            for blk in self.blocks:
                x = blk(x)

        x = self.norm(x)
        if self.fc_norm is not None:
            # me: add frame-level prediction support
            if self.keep_temporal_dim:
                x = rearrange(x, 'b (t hw) c -> b c t hw',
                              t=self.patch_embed.temporal_seq_len,
                              hw=self.patch_embed.spatial_num_patches)
                # spatial mean pooling
                x = x.mean(-1) # (B, C, T)
                # temporal upsample: 8 -> 16, for patch embedding reduction
                x = torch.nn.functional.interpolate(
                    x, scale_factor=self.patch_embed.tubelet_size,
                    mode='linear'
                )
                x = rearrange(x, 'b c t -> b t c')
                return self.fc_norm(x)
            else:
                return self.fc_norm(x.mean(1))
        else:
            # return x[:, 0]
            # NOTE: tmp change for mer2023 feature extraction in 2023/11/27 on 102!!!
            return x.mean(1)

    def forward(self, x, save_feature=False):
        x = self.forward_features(x)
        if save_feature:
            feature = x
        x = self.head(x)
        # me: add head activation function support
        x = self.head_activation_func(x)
        # me: add frame-level prediction support
        if self.keep_temporal_dim:
            x = x.view(x.size(0), -1) # (B,T,C) -> (B,T*C)
        if save_feature:
            return x, feature
        else:
            return x

@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

# me: for voxceleb2 pre-training
@register_model
def vit_base_patch16_112(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=112,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_tiny_dim192_patch16_160(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=160,
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_tiny_dim256_patch16_160(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=160,
        patch_size=16, embed_dim=256, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_small_half_dim_patch16_160(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=160,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_small_half_dim_patch16_192(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=192,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_small_half_dim_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

# me: no depth
@register_model
def vit_small_dim384_no_depth_patch16_160(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=160,
        patch_size=16, embed_dim=384, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

# me: no depth, copy from vit_small_dim384_no_depth_patch16_160
@register_model
def vit_small_dim256_no_depth_patch16_160(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=160,
        patch_size=16, embed_dim=256, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_small_half_depth_patch16_160(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=160,
        patch_size=16, embed_dim=768, depth=6, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_base_patch16_160(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=160,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_base_dim512_patch16_112(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=112,
        patch_size=16, embed_dim=512, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_base_dim512_patch16_160(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=160,
        patch_size=16, embed_dim=512, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_base_dim512_patch16_192(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=160,
        patch_size=16, embed_dim=512, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_base_dim512_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=160,
        patch_size=16, embed_dim=512, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_base_dim512_no_depth_patch16_160(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=160,
        patch_size=16, embed_dim=512, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_base_dim384_patch16_160(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=160,
        patch_size=16, embed_dim=384, depth=48, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_large_patch16_512(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
