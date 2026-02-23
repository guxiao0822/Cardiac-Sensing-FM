import numpy as np
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import torch.nn.functional as F

import random


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            dots_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(mask, 'b j -> b 1 1 j')
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(dots_mask, mask_value)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x, mask=None, return_intermediate=False):
        intermediates = []
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
            if return_intermediate:
                intermediates.append(x)
        if return_intermediate:
            return self.norm(x), intermediates
        return self.norm(x)



def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv1d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv1d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv1d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv1d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    scratch.layer_rn = nn.ModuleList([
        scratch.layer1_rn,
        scratch.layer2_rn,
        scratch.layer3_rn,
        scratch.layer4_rn,
    ])

    return scratch


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv1d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv1d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm1d(features)
            self.bn2 = nn.BatchNorm1d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

        # return out + x

def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv1d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="linear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

class CSFM(nn.Module):
    def __init__(self, *, signal_size, patch_size, text_len, num_classes, dim, depth, heads, mlp_dim, pool='cls',
                 channels=1, dim_head=64, dropout=0., emb_dropout=0., vocab_size=30522, output='dense'):
        super().__init__()
        signal_len = signal_size  # this is temporal length
        patch_size = patch_size  # this is patch size

        assert signal_len % patch_size == 0, 'Dimensions must be divisible by the patch size.'

        num_patches = signal_len // patch_size  # this is the num of patches of which channel
        patch_dim = patch_size  # this is the dimension of patches

        self.num_patches = num_patches
        self.num_text = text_len
        self.num_channels = channels
        self.encoder_dim = dim
        self.vocab_size = vocab_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.ts_to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) -> b c h p1', p1=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )  # time series projection embeddings

        self.text_to_embedding = nn.Sequential(
            nn.Embedding(vocab_size, dim),
            nn.LayerNorm(dim),
        )  # text projection embeddings

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.ts_channel_type_embedding = nn.Parameter(torch.randn(1, channels, dim))
        self.text_type_embedding = nn.Parameter(torch.randn(1, 1, dim))

        ## time series and text embeddings
        self.ts_pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.text_pos_embedding = nn.Parameter(torch.randn(1, text_len, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # Classification Head
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

        # Dense output Head (to be added)


    def forward(self, ts, channel, text=None, mask=None, task='cls'):
        """
        seq: [batch_size, channel, ts]
        channel: [channel_indices]
        text: [batch_size, text_len]
        mask: [batch_size, num_tokens]
        """

        ### Step 1 Process time series ###
        ts = self.ts_to_patch_embedding(ts)
        b, c, n, _ = ts.shape
        num_ts_tokens = c*n

        ts_channel_emb = repeat(self.ts_channel_type_embedding[:, channel], '1 c d -> b c n d', n=n, b=b)
        ts += ts_channel_emb

        ts_position_emb = repeat(self.ts_pos_embedding, '1 n d -> b c n d', c=c, b=b)
        ts += ts_position_emb

        ts = rearrange(ts, 'b c n d -> b (c n) d')

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)

        ### Step 2 Process text ###
        if text is not None:
            text = self.text_to_embedding(text)  # b x t x d
            text += self.text_pos_embedding
            text += self.text_type_embedding

            x = torch.cat((cls_tokens, ts, text), dim=1)

        else:
            x = torch.cat((cls_tokens, ts), dim=1)

        x = self.dropout(x)

        ### Step 3 Transformer ###
        x = self.transformer(x, mask=mask)

        ### Step 4 Classifier ###
        if task == 'cls':
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            x = self.to_latent(x)
            classification_output = self.mlp_head(x)

            return classification_output

        if task == 'dense':
            ### Dense Prediction ###
            # Hook decoder onto 4 layers from specified ViT layers
            layers = [layer[:, 1:1+num_ts_tokens] for layer in self.intermediate_features]

            # Reshape tokens to spatial representation
            layers = [rearrange(l, 'b (c n) d -> b (c d) n', n=n, c=c) for l in layers]

            # Postprocess activations
            layers = [self.dense_act_postprocess[idx](l) for idx, l in enumerate(layers)]

            # Project layers to chosen feature dim
            layers = [self.dense_scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

            # Fuse layers using refinement stages
            path_4 = self.dense_scratch.refinenet4(layers[3])
            path_3 = self.dense_scratch.refinenet3(path_4, layers[2])
            path_2 = self.dense_scratch.refinenet2(path_3, layers[1])
            path_1 = self.dense_scratch.refinenet1(path_2, layers[0])

            # Output head
            dense_output = self.dense_head(path_1)

            return dense_output

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        Specify the set of parameter names that should not be subjected to weight decay.
        """
        return {
            'cls_token',             # Exclude the class token from weight decay
            'ts_channel_type_embedding',  # Exclude the time series channel type embedding
            'text_type_embedding',   # Exclude the text type embedding
            'ts_pos_embedding',      # Exclude the time series positional embeddings
            'text_pos_embedding'     # Exclude the text positional embeddings
        }

class ChannelWiseLinear(nn.Module):
    def __init__(self, num_channels, in_features, out_features):
        super(ChannelWiseLinear, self).__init__()
        self.num_channels = num_channels
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(num_channels, in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(num_channels, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x, channel_indices):
        # x: [total_patches, in_features]
        # channel_indices: [total_patches]
        weight = self.weight[channel_indices]  # [total_patches, in_features, out_features]
        bias = self.bias[channel_indices]      # [total_patches, out_features]
        x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
        return x

def CSFM_model(variant='Base'):
    """
    Build CSFM model variants: 'Tiny', 'Base', or 'Large'.
    Only dim, depth, heads, and mlp_dim change across variants.
    """
    # Shared defaults
    base_args = dict(
        signal_size=2500,
        patch_size=25,
        num_classes=1,
        channels=13,
        dropout=0.1,
        emb_dropout=0.1,
        text_len=64,
        pool='cls',
    )

    # Variant-specific scaling
    variant = variant.lower()
    if variant == 'tiny':
        arch = dict(dim=768, depth=6, heads=8, mlp_dim=1024)
    elif variant == 'base':
        arch = dict(dim=768, depth=12, heads=12, mlp_dim=3072)
    elif variant == 'large':
        arch = dict(dim=1024, depth=16, heads=24, mlp_dim=4096)
    else:
        raise ValueError(f"Unknown CSFM variant '{variant}'. Use 'Tiny', 'Base', or 'Large'.")

    # Merge defaults + architecture parameters
    model = CSFM(**base_args, **arch)
    return model


if __name__ == '__main__':

 v = CSFM_model('Tiny')

 print(v)


