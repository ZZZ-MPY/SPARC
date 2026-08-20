   
                                             
                                                  
               
                          
   
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


                                                           
                            
                                                           
class CompositeCosineLinear(nn.Module):
    def __init__(self, in_features, sigma=12.0):
        super().__init__()
        self.in_features = in_features
        self.heads = nn.ModuleList([])
        self.sigmas = nn.ParameterList([])
        self.default_sigma = sigma

    def add_head(self, num_new_classes, device='cuda'):
        for h in self.heads:
            for p in h.parameters():
                p.requires_grad = False
        for s in self.sigmas:
            s.requires_grad = False

        new_head = nn.Linear(self.in_features, num_new_classes, bias=False).to(device)
        nn.init.kaiming_normal_(new_head.weight, a=math.sqrt(5))
        self.heads.append(new_head)

        new_sigma = nn.Parameter(
            torch.tensor(self.default_sigma, dtype=torch.float32, device=device),
            requires_grad=False)
        self.sigmas.append(new_sigma)

        print(f"[CosineHead] Added Head with {num_new_classes} classes, Constant Sigma={self.default_sigma:.2f}")

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)
        outputs = []
        for h, s in zip(self.heads, self.sigmas):
            w_norm = F.normalize(h.weight, p=2, dim=1, eps=1e-8)
            out = F.linear(x_norm, w_norm)
            out = s.abs() * out
            outputs.append(out)
        return torch.cat(outputs, dim=1)

    @property
    def out_features(self):
        return sum(h.out_features for h in self.heads)


                                                           
      
                                                           
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth'),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

                                       
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


                                                           
                      
                                                           
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer)
            for i in range(depth)])
        self.depth = depth
        self.norm = norm_layer(embed_dim)

        self.head = CompositeCosineLinear(embed_dim, sigma=12.0)
        if num_classes > 0:
            self.head.add_head(num_classes)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = CompositeCosineLinear(self.embed_dim, sigma=12.0)
        self.head.add_head(num_classes)

    def increment_classes(self, new_classes_count, device='cuda'):
        print(f"[SDT2Net] Incrementing classes with Per-Head Sigma strategy.")

        if isinstance(self.head, CompositeCosineLinear):
            self.head.add_head(new_classes_count, device)
        else:
            print("[Warning] Converting to CompositeCosineLinear...")
            old_head = self.head
            self.head = CompositeCosineLinear(self.embed_dim, sigma=12.0).to(device)

            if isinstance(old_head, nn.Linear):
                restored = nn.Linear(self.embed_dim, old_head.out_features, bias=False).to(device)
                restored.weight.data = old_head.weight.data.clone()
                for p in restored.parameters():
                    p.requires_grad = False
                self.head.heads.append(restored)
                self.head.sigmas.append(
                    nn.Parameter(torch.tensor(12.0, device=device), requires_grad=False))

            self.head.add_head(new_classes_count, device)

        self.num_classes = self.head.out_features
        print(f"[SDT2Net] Total classes: {self.num_classes}")

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for i in range(self.depth):
            x = self.blocks[i](x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x[:, 0])
        return x


                                                           
      
                                                           
def _conv_filter(state_dict, patch_size=16):
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384,
        depth=12, num_heads=6, mlp_ratio=4.,
        qkv_bias=True, **kwargs)
    model.default_cfg = default_cfgs['vit_small_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes,
            in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


@register_model
def vit_deit_SDT2Net_small_patch16_224(pretrained=False, **kwargs):
    return vit_small_patch16_224(pretrained=pretrained, **kwargs)