"""
models/backbones.py
===================================================================
Backbone cho CCKD-SSVEP (Cross-Channel / privileged-information distillation).

  - FBTCNN : port PyTorch trung thanh cua Ding et al., FB-tCNN (IEEE TNSRE 2021).
             Backbone CHINH.  Input: [B, n_bands, C, T].
  - TCNN   : FB-tCNN single-band (n_bands=1).  Input: [B, C, T].
  - EEGNet : compact CNN kiem chung (EEGNet-8,2).  Input: [B, C, T].

TINH CHAT MAU CHOT CHO KD:
  Conv dau tien co kernel (C, 1) (FB-tCNN/tCNN) hoac depthwise (C,1) (EEGNet)
  -> GOP toan bo kenh ve 1. Do do feature truoc classifier co so chieu KHONG
  phu thuoc so kenh C. => teacher 9C va student 3C/1C cho feature CUNG CHIEU,
  distillation (logit / similarity-preserving) chay truc tiep, khong can projection.

Interface khop project cua ban:
  model = build_backbone(name, n_channels, n_samples, n_classes, n_bands=3)
  logits          = model(x)
  logits, feat    = model(x, return_feat=True)   # feat: [B, feat_dim] cho KD

X tu loader cua ban la [N, C, T]. Voi FBTCNN, ap filter bank truoc:
  Xfb = apply_filter_bank(X, fs=250)   # -> [N, n_bands, C, T]
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================
# Filter bank (tien xu ly cho FB-tCNN) — chay 1 lan tren numpy X
# ===================================================================
def apply_filter_bank(X, fs=250, bands=((6, 50), (14, 50), (22, 50))):
    """[N, C, T] -> [N, n_bands, C, T] (float32).
    Ban goc FB-tCNN dung 4 sub-band; o day mac dinh 3 theo de xuat thiet ke.
    Doi 'bands' neu muon 4 dai."""
    from scipy.signal import butter, filtfilt
    N, C, T = X.shape
    out = np.empty((N, len(bands), C, T), dtype=np.float32)
    nyq = fs / 2.0
    for bi, (lo, hi) in enumerate(bands):
        b, a = butter(4, [lo / nyq, min(hi, nyq - 1) / nyq], btype="band")
        out[:, bi] = filtfilt(b, a, X, axis=-1).astype(np.float32)
    return out


class _SameConvStride(nn.Module):
    """Conv2d theo thoi gian, padding kieu TF-'same' + stride. Kernel (1, kw)."""
    def __init__(self, cin, cout, kw, stride):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, (1, kw), stride=(1, stride))
        self.kw, self.stride = kw, stride

    def forward(self, x):
        W = x.shape[-1]
        out_w = math.ceil(W / self.stride)
        pad = max((out_w - 1) * self.stride + self.kw - W, 0)
        return self.conv(F.pad(x, (pad // 2, pad - pad // 2)))


# ===================================================================
# FB-tCNN  (backbone chinh)
# ===================================================================
class FBTCNN(nn.Module):
    """
    Port trung thanh Ding et al. FB-tCNN. 3 conv (spatial-collapse, temporal-stride,
    temporal) DUNG CHUNG trong so giua cac sub-band; cong (add) roi conv4 gop thoi gian.
    Feature truoc classifier = 32-dim (doc lap voi C).
    """
    def __init__(self, n_channels, n_samples, n_classes, n_bands=3,
                 out_channel=16, lenth=5, drop=0.4):
        super().__init__()
        self.n_bands = n_bands
        self.expects_filterbank = (n_bands > 1)
        # 3 conv shared across bands
        self.conv1 = nn.Conv2d(1, out_channel, (n_channels, 1))            # gop kenh -> height 1
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = _SameConvStride(out_channel, out_channel, kw=n_samples, stride=5)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel, (1, lenth))       # valid
        self.bn3 = nn.BatchNorm2d(out_channel)
        # conv4 gop het chieu thoi gian con lai -> width 1
        w2 = math.ceil(n_samples / 5)
        w3 = w2 - lenth + 1
        assert w3 >= 1, f"n_samples={n_samples} qua ngan cho FB-tCNN"
        self.conv4 = nn.Conv2d(out_channel, 32, (1, w3))
        self.bn4 = nn.BatchNorm2d(32)
        self.drop = nn.Dropout(drop)
        self.feat_dim = 32
        self.classifier = nn.Linear(32, n_classes)

    def _branch(self, xb):                          # xb: [B,1,C,T]
        x = F.elu(self.bn1(self.conv1(xb)))
        x = self.drop(x)
        x = F.elu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        x = F.elu(self.bn3(self.conv3(x)))
        return x

    def forward(self, x, return_feat=False):
        if x.dim() == 3:                            # [B,C,T] -> 1 band
            x = x.unsqueeze(1)
        if self.n_bands > 1:
            assert x.shape[1] == self.n_bands, \
                f"FBTCNN can input [B,{self.n_bands},C,T], nhan {tuple(x.shape)} " \
                f"(dung apply_filter_bank truoc)"
        feats = [self._branch(x[:, b:b + 1]) for b in range(x.shape[1])]
        x = torch.stack(feats, 0).sum(0)            # fuse band (add), shared weights [B,16,1,w]
        feat_mid = x.flatten(1)                     # feature TRUNG GIAN giau (736-d @250), doc lap C
        x = self.drop(x)
        x = F.elu(self.bn4(self.conv4(x)))          # [B,32,1,1]
        feat = x.flatten(1)                         # [B,32] penultimate
        logits = self.classifier(self.drop(feat))
        # return_feat -> list feature cho SP da tang ([trung gian, penultimate])
        return (logits, [feat_mid, feat]) if return_feat else logits


# ===================================================================
# EEGNet-8,2  (backbone kiem chung)
# ===================================================================
class EEGNet(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes,
                 F1=8, D=2, F2=16, kern_length=64, drop=0.5):
        super().__init__()
        self.expects_filterbank = False
        self.n_bands = 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_length), padding=(0, kern_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),   # depthwise: gop kenh
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(drop),
        )
        # suy feat_dim bang forward gia (chinh xac, tranh sai so padding/pooling)
        with torch.no_grad():
            d = self._features(torch.zeros(1, 1, n_channels, n_samples))
            self.feat_dim = d.shape[1]
        self.classifier = nn.Linear(self.feat_dim, n_classes)

    def _features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x.flatten(1)

    def forward(self, x, return_feat=False):
        if x.dim() == 3:
            x = x.unsqueeze(1)                      # [B,1,C,T]
        feat = self._features(x)
        logits = self.classifier(feat)
        return (logits, [feat]) if return_feat else logits


# ===================================================================
# Factory
# ===================================================================
BACKBONE_REGISTRY = {
    "fbtcnn": lambda C, T, n, nb: FBTCNN(C, T, n, n_bands=nb),
    "tcnn":   lambda C, T, n, nb: FBTCNN(C, T, n, n_bands=1),
    "eegnet": lambda C, T, n, nb: EEGNet(C, T, n),
}


def build_backbone(name, n_channels, n_samples, n_classes, n_bands=3):
    name = name.lower()
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"backbone '{name}' khong co. Chon: {list(BACKBONE_REGISTRY)}")
    return BACKBONE_REGISTRY[name](n_channels, n_samples, n_classes, n_bands)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
