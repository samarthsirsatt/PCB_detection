"""GMO-DETR architecture, extracted verbatim from gmo_detr_full_pipeline.py.

Only the notebook's inline sanity-check cells and `!pip` magics were dropped; every
class body below is byte-identical to the pipeline's, so an experiment run here is the
same architecture as the published 12-class run. Registration into Ultralytics and the
model-YAML builder follow at the bottom (parameterised: nc, num_queries, P2 head).
"""
import math
import torch
import torch.nn as nn

def autopad(k, p=None, d=1):
    """Pad to 'same'. k=kernel, p=pad, d=dilation."""
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class Conv(nn.Module):
    """Standard Conv2d + BatchNorm + SiLU  (the ultralytics 'Conv')."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g,
                              dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    """Depthwise Conv (groups = gcd(c1, c2))."""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class GhostConv(nn.Module):
    """
    GhostConv (GhostNet, 2020) — the backbone STEM.
    Paper §2.1: applied ONLY at the input stage; 'cheap' ops reconstruct
    half the channels instead of full convolutions.
    """
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.primary = Conv(c1, c_, k, s, None, g, act=act)
        self.cheap = Conv(c_, c_, 5, 1, None, g=c_, act=act)

    def forward(self, x):
        y = self.primary(x)
        return torch.cat((y, self.cheap(y)), 1)

class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim of an (N, C, H, W) tensor."""
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(c, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class GatedCNNBlock(nn.Module):
    """
    Faithful MambaOut Gated CNN block (Yu & Wang, 2025) — what Fig 3c depicts:
    "gated fusion between two linear paths and a convolution branch".
    Paper Eqs 1-2 are compressed notation for it.
        x̂       = LayerNorm(x)
        g, i, c = split(fc1(x̂))                  # gate | identity | conv paths
        c       = DWConv(c)                       # convolution branch
        x       = fc2( act(g) ⊙ concat(i, c) )    # gated fusion + w3 proj
        out     = x + shortcut
    expansion_ratio / conv_ratio use MambaOut defaults; these are the first
    knobs to turn if the full-model param count misses 12.05M.
    """
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7,
                 conv_ratio=1.0, act_layer=nn.GELU):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Conv2d(dim, hidden * 2, 1)
        self.act = act_layer()
        conv_ch = int(conv_ratio * dim)
        self.split = (hidden, hidden - conv_ch, conv_ch)
        self.conv = nn.Conv2d(conv_ch, conv_ch, kernel_size,
                              padding=kernel_size // 2, groups=conv_ch)
        self.fc2 = nn.Conv2d(hidden, dim, 1)   # w3 output projection

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split, dim=1)
        c = self.conv(c)
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=1))
        return x + shortcut

class DMambaOut(nn.Module):
    """
    Dual-path (CSP-style) wrapper around Gated CNN blocks (Paper Fig 3b, Eqs 3-6):
        x1, x2 = Split(Conv1x1(x))     # x1 = shortcut, x2 = modeling seed
        y_i    = f_i(y_{i-1})          # chain of n Gated CNN blocks
        y      = Conv1x1( Concat(x1, y_1..y_n) )
    Ambiguity: Eq 4 literally seeds with x1 (leaving x2 unused) — reads as a
    typo. Sensible CSP topology is default; literal_seed=True reproduces the text.
    """
    def __init__(self, c, n=1, literal_seed=False):
        super().__init__()
        self.c_ = c // 2
        self.literal_seed = literal_seed
        self.cv1 = Conv(c, 2 * self.c_, 1, 1)
        self.blocks = nn.ModuleList(GatedCNNBlock(self.c_) for _ in range(n))
        self.cv2 = Conv((1 + n) * self.c_, c, 1, 1)

    def forward(self, x):
        x1, x2 = self.cv1(x).split((self.c_, self.c_), dim=1)
        seed = x1 if self.literal_seed else x2
        outs, y = [], seed
        for blk in self.blocks:
            y = blk(y)
            outs.append(y)
        return self.cv2(torch.cat([x1, *outs], dim=1))


class GMOBlock(nn.Module):
    """One downsampling Conv + k stacked DMambaOut modules (Paper §2.1)."""
    def __init__(self, c1, c2, k=1, stride=2, inner_n=1):
        super().__init__()
        self.down = Conv(c1, c2, 3, stride)
        self.aggr = nn.Sequential(*[DMambaOut(c2, n=inner_n) for _ in range(k)])

    def forward(self, x):
        return self.aggr(self.down(x))

class GMONet(nn.Module):
    """
    GhostConv stem (s2) + GMO-Block1..4.
    Paper channels: 128, 256, 384, 384   |   depths: 1, 1, 1, 3
    Neck taps the last three stages -> strides 8 / 16 / 32.
    """
    def __init__(self, in_ch=3, stem_ch=64,
                 widths=(128, 256, 384, 384),
                 depths=(1, 1, 1, 3)):
        super().__init__()
        self.stem = GhostConv(in_ch, stem_ch, k=3, s=2)
        c_prev = stem_ch
        self.stages = nn.ModuleList()
        for w, d in zip(widths, depths):
            self.stages.append(GMOBlock(c_prev, w, k=d, stride=2))
            c_prev = w
        self.out_channels = list(widths[1:])   # channels of the 3 tapped maps

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for st in self.stages:
            x = st(x)
            feats.append(x)
        return feats[1], feats[2], feats[3]     # S3, S4, S5

class GSConv(nn.Module):
    """Slim-neck GSConv (Li et al. 2024). Eager: parser prepends c1, YAML gives c2(+k,s)."""
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, g=1, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, g=c_, act=act)      # depthwise DSConv

    def forward(self, x):
        x1 = self.cv1(x)
        y = torch.cat((x1, self.cv2(x1)), 1)
        b, n, h, w = y.size()
        y = y.view(b, 2, n // 2, h, w).transpose(1, 2).contiguous().view(b, n, h, w)
        return y

class RepConv(nn.Module):
    """Reparameterizable conv (3x3 + 1x1 branches) with fuse_convs() so
    Ultralytics' model.fuse() works during final validation and inference."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, act=True):
        super().__init__()
        self.g, self.c1, self.c2 = g, c1, c2
        self.act = self.default_act if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward(self, x):
        return self.act(self.conv1(x) + self.conv2(x))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    def _pad_1x1_to_3x3(self, w):
        return None if w is None else torch.nn.functional.pad(w, [1, 1, 1, 1])

    def _fuse_bn(self, branch):
        if branch is None:
            return 0, 0
        conv, bn = branch.conv, branch.bn
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def get_equivalent_kernel_bias(self):
        k1, b1 = self._fuse_bn(self.conv1)
        k2, b2 = self._fuse_bn(self.conv2)
        return k1 + self._pad_1x1_to_3x3(k2), b1 + b2

    def fuse_convs(self):
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(self.conv1.conv.in_channels, self.conv1.conv.out_channels,
                              kernel_size=3, stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding, groups=self.g,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for p in self.parameters():
            p.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")


class RepNBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = RepConv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepNCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0)
                                 for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class CAA(nn.Module):
    """
    Context Anchor Attention — PKINet-faithful (Cai et al. 2024, Eq 5 / Fig 2e).
    AvgPool -> 1x1 (reduce to C/r) -> horizontal DW (1xk) -> vertical DW (kx1)
    -> 1x1 (expand to C) -> sigmoid -> multiply. Strip convs run in the reduced
    space (r), matching PKINet; kernel length h_k/v_k default 11 (paper doesn't
    print exact values, so tunable).
    """
    def __init__(self, ch, reduction=4, h_kernel=11, v_kernel=11):
        super().__init__()
        rc = max(ch // reduction, 8)
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, rc, 1)                                  # reduce
        self.h_conv = nn.Conv2d(rc, rc, (1, h_kernel), 1,
                                (0, h_kernel // 2), groups=rc)        # horizontal strip
        self.v_conv = nn.Conv2d(rc, rc, (v_kernel, 1), 1,
                                (v_kernel // 2, 0), groups=rc)        # vertical strip
        self.conv2 = Conv(rc, ch, 1)                                  # expand
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn = self.conv1(self.avg_pool(x))
        attn = self.h_conv(attn)
        attn = self.v_conv(attn)
        attn = self.act(self.conv2(attn))
        return x * attn

class CAFF(nn.Module):
    """Context-Aware Feature Fusion — replaces RepC3. Eager: parser prepends c1."""
    def __init__(self, c1, c2, c3=None, c4=None, n=1):
        super().__init__()
        c3 = c3 or c2
        c4 = c4 or c2 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.caa = CAA(c3 + 2 * c4)
        self.cv4 = Conv(c3 + 2 * c4, c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        cat = self.caa(torch.cat(y, 1))
        return self.cv4(cat)

from einops import rearrange

class TSSA(nn.Module):
    """
    Token Statistics Self-Attention — ported faithfully from the official
    ToST implementation (Wu et al., ICLR 2025; the paper's ref [20]).
    Linear-complexity: no N×N similarity matrix. Operates on (B, N, C) tokens.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.heads = num_heads
        self.attend = nn.Softmax(dim=1)
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.temp = nn.Parameter(torch.ones(num_heads, 1))
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):                                  # x: (B, N, C)
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)
        w_normed = torch.nn.functional.normalize(w, dim=-2)      # normalize over tokens
        Pi = self.attend(torch.sum(w_normed ** 2, dim=-1) * self.temp)  # (B,h,N)
        dots = torch.matmul(
            (Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2),
            w ** 2)
        attn = 1.0 / (1 + dots)
        out = -torch.mul(w.mul(Pi.unsqueeze(-1)), attn)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

import math

def build_2d_sincos_pos_embed(w, h, dim, temperature=10000.0, device=None, dtype=None):
    """2D sine-cosine positional embedding, RT-DETR/AIFI convention. -> (1, w*h, dim)"""
    assert dim % 4 == 0, "embed dim must be divisible by 4 for 2D sin-cos"
    gx = torch.arange(w, device=device, dtype=torch.float32)
    gy = torch.arange(h, device=device, dtype=torch.float32)
    gx, gy = torch.meshgrid(gx, gy, indexing='ij')
    pos_dim = dim // 4
    omega = torch.arange(pos_dim, device=device, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature ** omega)
    ox = gx.flatten()[..., None] @ omega[None]
    oy = gy.flatten()[..., None] @ omega[None]
    pe = torch.cat([ox.sin(), ox.cos(), oy.sin(), oy.cos()], dim=1)[None]
    return pe.to(dtype=dtype) if dtype is not None else pe

class TAIFI(nn.Module):
    """
    Token-Aware Interaction with Feature Integration (Paper §2.4).
    Same wrapper as RT-DETR's AIFI (flatten -> +pos-embed -> token op -> FFN ->
    reshape), but MHSA is replaced by TSSA. Applied to the deepest map (S5).
    Post-norm Transformer block, matching the paper's description.
    """
    def __init__(self, c, num_heads=8, ffn_ratio=4, dropout=0.0, act=nn.GELU):
        super().__init__()
        self.tssa = TSSA(c, num_heads=num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(c, c * ffn_ratio), act(),
            nn.Dropout(dropout), nn.Linear(c * ffn_ratio, c))
        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                  # x: (B, C, H, W)
        b, c, h, w = x.shape
        pos = build_2d_sincos_pos_embed(w, h, c, device=x.device, dtype=x.dtype)
        src = x.flatten(2).permute(0, 2, 1)                # (B, HW, C)
        # post-norm: sublayer -> residual -> norm
        src = self.norm1(src + self.drop(self.tssa(src + pos)))
        src = self.norm2(src + self.drop(self.ffn(src)))
        return src.permute(0, 2, 1).view(b, c, h, w).contiguous()


# ---------------------------------------------------------------------------
# Registration into Ultralytics (pipeline CELL A, verbatim logic)
# ---------------------------------------------------------------------------
import re
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as ult_modules


def _register(mapping):
    for name, cls in mapping.items():
        setattr(ult_modules, name, cls)          # resolves via ultralytics.nn.modules
        setattr(tasks, name, cls)                # resolves via parse_model globals()
        for attr in ("base_modules", "global_modules"):
            s = getattr(tasks, attr, None)
            if isinstance(s, (set, frozenset)) and cls not in s:
                setattr(tasks, attr, frozenset(s | {cls}))


class GMONetBackbone(GMONet):
    """GMONet returning [S3, S4, S5] so one YAML layer emits all taps."""

    def forward(self, x):
        s3, s4, s5 = super().forward(x)
        return [s3, s4, s5]


class GMONetBackboneP2(GMONet):
    """Same backbone, but also emitting the stride-4 tap: [S2, S3, S4, S5].

    S2 is the first GMOBlock's output (128ch, stride 4) — already computed by the
    baseline backbone and simply discarded, so this adds no backbone parameters.
    """

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for st in self.stages:
            x = st(x)
            feats.append(x)
        return list(feats)


_CUSTOM = {
    "GMONet": GMONet, "GMOBlock": GMOBlock, "DMambaOut": DMambaOut,
    "GatedCNNBlock": GatedCNNBlock, "LayerNorm2d": LayerNorm2d,
    "GSConv": GSConv, "CAFF": CAFF, "CAA": CAA,
    "RepNCSP": RepNCSP, "RepConv": RepConv, "RepNBottleneck": RepNBottleneck,
    "TAIFI": TAIFI, "TSSA": TSSA,
    "GMONetBackbone": GMONetBackbone, "GMONetBackboneP2": GMONetBackboneP2,
}

_PATCHED = False


def register():
    """Register the custom modules and re-patch parse_model to accept CAFF/GSConv."""
    global _PATCHED
    if _PATCHED:
        return
    _register(_CUSTOM)

    with open(tasks.__file__) as f:
        lines = f.read().splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if l.startswith("def parse_model"))
    end = start + 1
    while end < len(lines):
        l = lines[end]
        if l[:1].strip() and (l.startswith(("def ", "class ", "@"))):
            break
        end += 1
    func_src = "".join(lines[start:end])
    m = re.search(r"base_modules\s*=\s*frozenset\(\s*\{\s*\n", func_src)
    assert m, "frozenset opening not found in source"
    inject = m.group(0) + "            CAFF,\n            GSConv,\n"
    patched = func_src[:m.start()] + inject + func_src[m.end():]
    tasks.CAFF = CAFF
    tasks.GSConv = GSConv
    exec(patched, tasks.__dict__)
    _PATCHED = True
    print("[gmo] registered:", ", ".join(sorted(_CUSTOM)))


# ---------------------------------------------------------------------------
# Model YAML — the pipeline's gmo_detr_4a, parameterised
# ---------------------------------------------------------------------------
_HEAD_3 = """
backbone:
  - [-1, 1, GMONetBackbone, []]      # 0  -> [S3(256), S4(384), S5(384)]
  - [0, 1, Index, [256, 0]]          # 1  P3/8   256ch
  - [0, 1, Index, [384, 1]]          # 2  P4/16  384ch
  - [0, 1, Index, [384, 2]]          # 3  P5/32  384ch

head:
  - [3, 1, Conv, [256, 1, 1, None, 1, 1, False]]  # 4 input_proj.2
  - [-1, 1, TAIFI, [256, 8]]                        # 5 TAIFI
  - [-1, 1, GSConv, [256, 1, 1]]                    # 6 Y5

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]     # 7
  - [2, 1, Conv, [256, 1, 1, None, 1, 1, False]]   # 8 input_proj.1
  - [[-2, -1], 1, Concat, [1]]                      # 9
  - [-1, 1, CAFF, [256]]                            # 10 F4
  - [-1, 1, GSConv, [256, 1, 1]]                    # 11 Y4

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]     # 12
  - [1, 1, Conv, [256, 1, 1, None, 1, 1, False]]   # 13 input_proj.0
  - [[-2, -1], 1, Concat, [1]]                      # 14
  - [-1, 1, CAFF, [256]]                            # 15 X3 (P3 out)

  - [-1, 1, GSConv, [256, 3, 2]]                    # 16
  - [[-1, 11], 1, Concat, [1]]                      # 17
  - [-1, 1, CAFF, [256]]                            # 18 (P4 out)

  - [-1, 1, GSConv, [256, 3, 2]]                    # 19
  - [[-1, 6], 1, Concat, [1]]                       # 20
  - [-1, 1, CAFF, [256]]                            # 21 (P5 out)

  - [[15, 18, 21], 1, RTDETRDecoder, [nc, 256, {nq}]]   # 22
"""

_HEAD_4 = """
backbone:
  - [-1, 1, GMONetBackboneP2, []]    # 0  -> [S2(128), S3(256), S4(384), S5(384)]
  - [0, 1, Index, [128, 0]]          # 1  P2/4   128ch
  - [0, 1, Index, [256, 1]]          # 2  P3/8   256ch
  - [0, 1, Index, [384, 2]]          # 3  P4/16  384ch
  - [0, 1, Index, [384, 3]]          # 4  P5/32  384ch

head:
  - [4, 1, Conv, [256, 1, 1, None, 1, 1, False]]   # 5 input_proj P5
  - [-1, 1, TAIFI, [256, 8]]                        # 6 TAIFI
  - [-1, 1, GSConv, [256, 1, 1]]                    # 7 Y5

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]     # 8
  - [3, 1, Conv, [256, 1, 1, None, 1, 1, False]]   # 9 input_proj P4
  - [[-2, -1], 1, Concat, [1]]                      # 10
  - [-1, 1, CAFF, [256]]                            # 11 F4
  - [-1, 1, GSConv, [256, 1, 1]]                    # 12 Y4

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]     # 13
  - [2, 1, Conv, [256, 1, 1, None, 1, 1, False]]   # 14 input_proj P3
  - [[-2, -1], 1, Concat, [1]]                      # 15
  - [-1, 1, CAFF, [256]]                            # 16 F3
  - [-1, 1, GSConv, [256, 1, 1]]                    # 17 Y3

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]     # 18
  - [1, 1, Conv, [256, 1, 1, None, 1, 1, False]]   # 19 input_proj P2
  - [[-2, -1], 1, Concat, [1]]                      # 20
  - [-1, 1, CAFF, [256]]                            # 21 X2 (P2 out, stride 4)

  - [-1, 1, GSConv, [256, 3, 2]]                    # 22
  - [[-1, 17], 1, Concat, [1]]                      # 23
  - [-1, 1, CAFF, [256]]                            # 24 (P3 out)

  - [-1, 1, GSConv, [256, 3, 2]]                    # 25
  - [[-1, 12], 1, Concat, [1]]                      # 26
  - [-1, 1, CAFF, [256]]                            # 27 (P4 out)

  - [-1, 1, GSConv, [256, 3, 2]]                    # 28
  - [[-1, 7], 1, Concat, [1]]                       # 29
  - [-1, 1, CAFF, [256]]                            # 30 (P5 out)

  - [[21, 24, 27, 30], 1, RTDETRDecoder, [nc, 256, {nq}]]   # 31
"""


def make_yaml(dest, nc=8, nq=300, p2=False):
    """Write the GMO-DETR model YAML. nq=300 is the RTDETRDecoder default."""
    from pathlib import Path
    head = (_HEAD_4 if p2 else _HEAD_3).format(nq=nq)
    txt = f"nc: {nc}\nscales:\n  s: [1.00, 1.00, 2048]\n{head}"
    dest = Path(dest)
    dest.write_text(txt)
    return dest
