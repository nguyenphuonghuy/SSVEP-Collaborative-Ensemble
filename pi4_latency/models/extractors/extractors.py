"""
models/extractors/extractors.py
================================
4 Feature Extractors cho SSVEP Ensemble (theo handoff doc)
- Extractor 1: Temporal CNN (~6K params)
- Extractor 2: Frequency-Domain CNN (~6K params)
- Extractor 3: Filter-Bank CNN / FBCCA-inspired (~7K params)
- Extractor 4: Spatial-Temporal CNN (~6K params)
Target: mỗi extractor ~6-7K params, competitive với EEGNet baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# BASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class BaseExtractor(nn.Module):
    """Base class cho tất cả extractors."""
    
    def __init__(self, n_channels: int, n_samples: int, n_classes: int):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
    
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_feature_dim(self) -> int:
        """Return feature embedding dimension (before classifier)."""
        raise NotImplementedError
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass → feature embedding (trước classifier)."""
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass → logits."""
        features = self.forward_features(x)
        return self.classifier(features)


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTOR 1: Temporal CNN (EEGNet-inspired)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalCNN(BaseExtractor):
    """
    Temporal CNN - EEGNet-inspired, ~6K params.
    
    Tập trung vào temporal patterns trong EEG:
    - Depthwise temporal convolutions
    - Spatial filtering
    - Compact feature representation
    
    Input: (batch, n_channels, n_samples)
    Output: (batch, n_classes)
    """
    
    def __init__(
        self,
        n_channels: int = 9,
        n_samples: int = 125,
        n_classes: int = 40,
        F1: int = 8,           # temporal filters
        D: int = 2,            # depth multiplier
        F2: int = 16,          # pointwise filters
        dropout: float = 0.3
    ):
        super().__init__(n_channels, n_samples, n_classes)
        
        # Block 1: Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(F1),
        )
        
        # Block 2: Depthwise spatial convolution
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        
        # Block 3: Separable convolution
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 15), padding=(0, 7),
                      groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        
        # Compute feature dimension
        self._feature_dim = self._compute_feature_dim()
        self.classifier = nn.Linear(self._feature_dim, n_classes)

    def _compute_feature_dim(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self.temporal_conv(x)
            x = self.spatial_conv(x)
            x = self.separable_conv(x)
            return x.flatten(1).shape[1]
    
    def get_feature_dim(self) -> int:
        return self._feature_dim
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_channels, n_samples)
        x = x.unsqueeze(1)  # (batch, 1, n_channels, n_samples)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        return x.flatten(1)


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTOR 2: Frequency-Domain CNN
# ─────────────────────────────────────────────────────────────────────────────

class FreqDomainCNN(BaseExtractor):
    """
    SSVEPFreqNet — Frequency-Domain CNN được thiết kế lại cho 4s window.

    Vấn đề của kiến trúc cũ với 4s:
    - Hardcode 64 FFT bins → bỏ qua hầu hết thông tin tần số
    - CNN quá nhỏ, không phù hợp input dài

    Thiết kế mới — khai thác đúng thế mạnh của 4s FFT:
    1. FFT 4s → resolution 0.25 Hz/bin → phân biệt được 40 targets cách nhau 0.2Hz
    2. Chỉ lấy SSVEP band (6–90Hz) → giảm noise, compact representation
    3. Harmonic stacking: stack amplitude tại f, 2f, 3f, 4f, 5f cho mỗi target
       → explicit SSVEP feature, không cần CNN học từ đầu
    4. Spatial weighting qua Conv2d trên channel dimension

    Input:  (batch, n_channels, n_samples)
    Output: (batch, n_classes)
    """

    def __init__(
        self,
        n_channels: int = 9,
        n_samples: int = 1000,   # 4s @ 250Hz
        n_classes: int = 40,
        sfreq: int = 250,
        n_harmonics: int = 5,    # số harmonics dùng
        dropout: float = 0.3,
    ):
        super().__init__(n_channels, n_samples, n_classes)
        self.sfreq = sfreq
        self.n_harmonics = n_harmonics

        # Tần số 40 SSVEP targets (Benchmark dataset)
        self.register_buffer('target_freqs', torch.tensor([
            8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2,
            8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4,
            8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6,
            8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8,
        ], dtype=torch.float32))  # (40,)

        # Input cho CNN: (batch, n_channels, n_classes, n_harmonics)
        # Spatial + class + harmonic fusion
        self.spatial_conv = nn.Sequential(
            # Pool across channels
            nn.Conv2d(n_channels, 8, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.class_conv = nn.Sequential(
            # Pattern across harmonics per class
            nn.Conv2d(8, 16, kernel_size=(1, n_harmonics), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self._feature_dim = self._compute_feature_dim()
        self.classifier = nn.Linear(self._feature_dim, n_classes)

    def _get_harmonic_amplitudes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tính amplitude tại các harmonic frequencies của 40 targets.

        Returns: (batch, n_channels, n_classes, n_harmonics)
        """
        batch = x.shape[0]

        # FFT
        fft = torch.fft.rfft(x, dim=-1)           # (batch, n_ch, n_fft)
        amp = torch.abs(fft)                        # (batch, n_ch, n_fft)

        # Frequency resolution
        freq_res = self.sfreq / self.n_samples      # Hz per bin
        n_fft = amp.shape[-1]

        # Lấy amplitude tại mỗi harmonic của mỗi target
        # target_freqs: (40,), harmonics: 1..n_harmonics
        harmonic_amps = []
        for h in range(1, self.n_harmonics + 1):
            freqs_h = self.target_freqs * h         # (40,) harmonic frequencies
            bins_h = torch.floor(freqs_h / freq_res + 0.5).long().clamp(0, n_fft - 1)  # (40,) bin indices
            # amp[:, :, bins_h]: (batch, n_ch, 40)
            harmonic_amps.append(amp[:, :, bins_h])

        # Stack: (batch, n_ch, 40, n_harmonics)
        return torch.stack(harmonic_amps, dim=-1)

    def _compute_feature_dim(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, self.n_channels, self.n_samples)
            feat = self._extract(x)
            return feat.shape[1]

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, n_ch, 40, n_harmonics)
        h = self._get_harmonic_amplitudes(x)

        # Normalize per sample
        h = h / (h.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)

        # Spatial conv: (batch, n_ch, 40, n_harm) → (batch, 8, 40, n_harm)
        out = self.spatial_conv(h)

        # Class conv: (batch, 8, 40, n_harm) → (batch, 16, 40, 1)
        out = self.class_conv(out)

        return out.flatten(1)   # (batch, 16*40)

    def get_feature_dim(self) -> int:
        return self._feature_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._extract(x)


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE ABLATION: tach bach dong gop cua "harmonic stacking" (bieu dien)
# khoi dong gop cua CNN co the hoc duoc (novelty ablation). Hai
# lop duoi day co CUNG tien xu ly tan so voi FreqDomainCNN nhung thay 2 khoi
# CNN (spatial_conv + class_conv) bang MOT lop tuyen tinh duy nhat.
# ─────────────────────────────────────────────────────────────────────────────

_BENCHMARK_TARGET_FREQS = [
    8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2,
    8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4,
    8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6,
    8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8,
]


class HarmonicAmpLinear(BaseExtractor):
    """
    Baseline 1: chi lay bien do FFT tai TAN SO CO BAN (khong xep chong hai)
    cua moi target, dua thang qua MOT lop tuyen tinh. Khong co cau truc
    tensor "harmonic stacking", khong co CNN.

    Muc dich: kiem tra xem rieng thong tin bien do tan so co ban da du de
    phan biet cac target hay chua, truoc khi them hai bac cao va CNN.
    """

    def __init__(
        self,
        n_channels: int = 9,
        n_samples: int = 1000,
        n_classes: int = 40,
        sfreq: int = 250,
        dropout: float = 0.3,
    ):
        super().__init__(n_channels, n_samples, n_classes)
        self.sfreq = sfreq
        self.register_buffer(
            'target_freqs',
            torch.tensor(_BENCHMARK_TARGET_FREQS[:n_classes], dtype=torch.float32)
        )
        self._feature_dim = n_channels * n_classes
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self._feature_dim, n_classes)

    def _get_fundamental_amplitude(self, x: torch.Tensor) -> torch.Tensor:
        """Returns: (batch, n_channels, n_classes) - bien do tai f_n, khong co hai."""
        fft = torch.fft.rfft(x, dim=-1)
        amp = torch.abs(fft)
        freq_res = self.sfreq / self.n_samples
        n_fft = amp.shape[-1]
        bins = torch.floor(self.target_freqs / freq_res + 0.5).long().clamp(0, n_fft - 1)
        return amp[:, :, bins]

    def get_feature_dim(self) -> int:
        return self._feature_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        a = self._get_fundamental_amplitude(x)
        a = a / (a.amax(dim=(1, 2), keepdim=True) + 1e-8)
        a = self.dropout(a)
        return a.flatten(1)


class HarmonicStackLinear(BaseExtractor):
    """
    Baseline 2: dung DUNG tensor harmonic-stacking (C x N x H, 5 harmonics)
    y het FreqDomainCNN, nhung dua thang qua MOT lop tuyen tinh thay vi 2
    khoi CNN (spatial_conv + class_conv).

    Muc dich: tach bach dong gop cua BIEU DIEN xep chong hai (representation)
    khoi dong gop cua CNN co the hoc duoc. Neu baseline nay da manh gan
    bang FreqDomainCNN thi dong gop chinh la bieu dien, con neu FreqDomainCNN
    vuot troi ro ret thi dong gop chinh nam o phan CNN hoc bieu dien tu du
    lieu (learnable harmonic representation), khong phai ban than thao tac
    xep chong.
    """

    def __init__(
        self,
        n_channels: int = 9,
        n_samples: int = 1000,
        n_classes: int = 40,
        sfreq: int = 250,
        n_harmonics: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__(n_channels, n_samples, n_classes)
        self.sfreq = sfreq
        self.n_harmonics = n_harmonics
        self.register_buffer(
            'target_freqs',
            torch.tensor(_BENCHMARK_TARGET_FREQS[:n_classes], dtype=torch.float32)
        )
        self._feature_dim = n_channels * n_classes * n_harmonics
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self._feature_dim, n_classes)

    def _get_harmonic_amplitudes(self, x: torch.Tensor) -> torch.Tensor:
        """Returns: (batch, n_channels, n_classes, n_harmonics)."""
        fft = torch.fft.rfft(x, dim=-1)
        amp = torch.abs(fft)
        freq_res = self.sfreq / self.n_samples
        n_fft = amp.shape[-1]
        harmonic_amps = []
        for h in range(1, self.n_harmonics + 1):
            freqs_h = self.target_freqs * h
            bins_h = torch.floor(freqs_h / freq_res + 0.5).long().clamp(0, n_fft - 1)
            harmonic_amps.append(amp[:, :, bins_h])
        return torch.stack(harmonic_amps, dim=-1)

    def get_feature_dim(self) -> int:
        return self._feature_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self._get_harmonic_amplitudes(x)
        h = h / (h.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
        h = self.dropout(h)
        return h.flatten(1)


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTOR 3: ShallowConvNet (thay thế FilterBankCNN)
# Paper: Schirrmeister et al. 2017 "Deep learning with convolutional neural
#        networks for EEG decoding and visualization"
# ─────────────────────────────────────────────────────────────────────────────

class ShallowConvNet(BaseExtractor):
    """
    ShallowConvNet - Schirrmeister 2017, adapted cho SSVEP 40-class, ~6K params.

    Ý tưởng cốt lõi:
    - Temporal conv kernel DÀI (25 samples = 100ms) → học oscillation patterns
      trực tiếp trên raw EEG mà không cần FFT, không bị giới hạn frequency resolution
    - Spatial conv → pool across channels (như beamforming)
    - Square + Log activation → non-linear power estimation
      (tương đương với log band power, feature chuẩn của SSVEP)
    - AvgPool temporal → compress

    Khác biệt với Ext1 (EEGNet):
    - EEGNet dùng depthwise separable conv, nhỏ hơn, generalizable hơn
    - ShallowConvNet dùng temporal conv dài hơn + power nonlinearity
    - Hai kiến trúc học feature khác nhau → diversity cho ensemble

    Input:  (batch, n_channels, n_samples)   raw EEG
    Output: (batch, n_classes)
    """

    def __init__(
        self,
        n_channels: int = 9,
        n_samples: int = 125,
        n_classes: int = 40,
        n_temporal_filters: int = 20,   # số temporal filters
        temporal_kernel: int = 25,       # ~100ms tại 250Hz
        pool_size: int = 15,             # AvgPool window
        pool_stride: int = 5,            # AvgPool stride
        dropout: float = 0.5,            # ShallowConvNet gốc dùng 0.5
    ):
        super().__init__(n_channels, n_samples, n_classes)
        self.n_temporal_filters = n_temporal_filters
        self.temporal_kernel = temporal_kernel
        self.dropout = dropout

        # Block 1: Temporal convolution
        # Học oscillatory patterns (sin waves ở SSVEP frequencies)
        # kernel_size=(1, 25): chỉ convolve theo time axis
        self.temporal_conv = nn.Conv2d(
            in_channels=1,
            out_channels=n_temporal_filters,
            kernel_size=(1, temporal_kernel),
            bias=False
        )

        # Block 2: Spatial convolution
        # Pool across all EEG channels → spatial weighting
        # kernel_size=(n_channels, 1): collapse channel dimension
        self.spatial_conv = nn.Conv2d(
            in_channels=n_temporal_filters,
            out_channels=n_temporal_filters,
            kernel_size=(n_channels, 1),
            bias=False
        )
        self.bn = nn.BatchNorm2d(n_temporal_filters, momentum=0.1, eps=1e-5)

        # Block 3: Square + Log nonlinearity (power estimation)
        # Không có learnable params, thực hiện trong forward()

        # Block 4: AvgPool temporal
        self.pool = nn.AvgPool2d(
            kernel_size=(1, pool_size),
            stride=(1, pool_stride)
        )
        self.drop = nn.Dropout(dropout)

        # Compute feature dim và build classifier
        self._feature_dim = self._compute_feature_dim()
        self.classifier = nn.Linear(self._feature_dim, n_classes)

    def _compute_feature_dim(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, self.n_channels, self.n_samples)
            feat = self._extract(x)
            return feat.shape[1]

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        """Core feature extraction."""
        # x: (batch, n_channels, n_samples)
        x = x.unsqueeze(1)                    # (batch, 1, n_channels, n_samples)

        x = self.temporal_conv(x)             # (batch, F, n_channels, T-kernel+1)
        x = self.spatial_conv(x)              # (batch, F, 1, T')
        x = self.bn(x)

        # Square + Log power nonlinearity (clamped để tránh log(0))
        x = torch.clamp(x ** 2, min=1e-6)
        x = torch.log(x)

        x = self.pool(x)                      # (batch, F, 1, T'')
        x = self.drop(x)
        return x.flatten(1)                   # (batch, F * T'')

    def get_feature_dim(self) -> int:
        return self._feature_dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._extract(x)


# ─────────────────────────────────────────────────────────────────────────────
# [Deprecated] FilterBankCNN — giữ lại để reference, không dùng trong ensemble
# ─────────────────────────────────────────────────────────────────────────────

class FilterBankCNN(BaseExtractor):
    """
    DEPRECATED: Thay thế bằng ShallowConvNet.
    FFT masking với 0.5s window (resolution 2Hz/bin) không đủ để phân biệt
    40 SSVEP targets cách nhau 0.2Hz. Mean LOSO chỉ đạt 26%.
    Giữ lại để reference/ablation study.
    """

    def __init__(self, n_channels=9, n_samples=125, n_classes=40,
                 n_bands=5, dropout=0.3, sfreq=250):
        super().__init__(n_channels, n_samples, n_classes)
        self.n_bands = n_bands
        self.sfreq = sfreq
        self.band_ranges = [(6,14),(14,22),(22,30),(30,38),(38,46)][:n_bands]
        self.band_encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(4), nn.ELU(),
            nn.Conv2d(4, 4, kernel_size=(1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(4), nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)), nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(n_bands*4, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(16), nn.ELU(),
            nn.AdaptiveAvgPool1d(8), nn.Dropout(dropout),
        )
        self._feature_dim = self._compute_feature_dim()
        self.classifier = nn.Linear(self._feature_dim, n_classes)

    def _compute_feature_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, self.n_channels, self.n_samples)
            bands = self._apply_filterbank(x)
            feats = []
            for b in bands:
                f = self.band_encoder(b.unsqueeze(1)).squeeze(2)
                feats.append(f)
            return self.fusion(torch.cat(feats, dim=1)).flatten(1).shape[1]

    def _apply_filterbank(self, x):
        fft = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(self.n_samples, 1.0/self.sfreq)
        bands = []
        for (low, high) in self.band_ranges:
            mask = ((freqs >= low) & (freqs <= high)).float()
            bands.append(torch.fft.irfft(fft * mask.to(fft.device),
                                          n=self.n_samples, dim=-1))
        return bands

    def get_feature_dim(self): return self._feature_dim

    def forward_features(self, x):
        feats = [self.band_encoder(b.unsqueeze(1)).squeeze(2)
                 for b in self._apply_filterbank(x)]
        return self.fusion(torch.cat(feats, dim=1)).flatten(1)


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTOR 4: Spatial-Temporal CNN
# ─────────────────────────────────────────────────────────────────────────────

class SpatialTemporalCNN(BaseExtractor):
    """
    Spatial-Temporal CNN - TRCA-inspired, ~6K params.
    
    Khai thác spatial correlation giữa các EEG channels:
    - Learned spatial filters (như TRCA)
    - Temporal convolutions sau spatial filtering
    - Compact weighted average output
    
    Input: (batch, n_channels, n_samples)
    Output: (batch, n_classes)
    """
    
    def __init__(
        self,
        n_channels: int = 9,
        n_samples: int = 125,
        n_classes: int = 40,
        n_spatial: int = 8,    # số spatial filters
        dropout: float = 0.3
    ):
        super().__init__(n_channels, n_samples, n_classes)
        
        # Spatial filtering (như TRCA inter-trial covariance)
        self.spatial_filter = nn.Sequential(
            nn.Conv2d(1, n_spatial, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(n_spatial),
            nn.ELU(),
        )
        
        # Temporal processing sau spatial
        self.temporal = nn.Sequential(
            nn.Conv2d(n_spatial, n_spatial, kernel_size=(1, 25), padding=(0, 12),
                      groups=n_spatial, bias=False),
            nn.BatchNorm2d(n_spatial),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(n_spatial, 16, kernel_size=(1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        
        self._feature_dim = self._compute_feature_dim()
        self.classifier = nn.Linear(self._feature_dim, n_classes)
    
    def _compute_feature_dim(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self.spatial_filter(x)
            x = self.temporal(x)
            return x.flatten(1).shape[1]
    
    def get_feature_dim(self) -> int:
        return self._feature_dim
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (batch, 1, n_channels, n_samples)
        x = self.spatial_filter(x)
        x = self.temporal(x)
        return x.flatten(1)


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY & UTILS
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTOR_REGISTRY = {
    'temporal_cnn': TemporalCNN,
    'freq_domain_cnn': FreqDomainCNN,
    'shallow_conv_net': ShallowConvNet,     # Ext3: thay FilterBankCNN
    'filter_bank_cnn': FilterBankCNN,       # Deprecated
    'spatial_temporal_cnn': SpatialTemporalCNN,
    # Baseline ablation (novelty cua harmonic stacking)
    'harmonic_amp_linear':   HarmonicAmpLinear,
    'harmonic_stack_linear': HarmonicStackLinear,
}


def build_extractor(
    name: str,
    n_channels: int = 9,
    n_samples: int = 125,
    n_classes: int = 40,
    **kwargs
) -> BaseExtractor:
    """Factory function để tạo extractor theo tên."""
    if name not in EXTRACTOR_REGISTRY:
        raise ValueError(f"Unknown extractor: {name}. "
                         f"Available: {list(EXTRACTOR_REGISTRY.keys())}")
    return EXTRACTOR_REGISTRY[name](n_channels, n_samples, n_classes, **kwargs)


def build_all_extractors(
    n_channels: int = 9,
    n_samples: int = 125,
    n_classes: int = 40
) -> dict:
    """Tạo tất cả 4 extractors."""
    return {
        name: build_extractor(name, n_channels, n_samples, n_classes)
        for name in EXTRACTOR_REGISTRY
    }


def count_params_summary(extractors: dict) -> None:
    """In bảng tóm tắt params của tất cả extractors."""
    print("\n" + "="*55)
    print(f"{'Extractor':<25} {'Params':>10} {'Feature Dim':>12}")
    print("="*55)
    total = 0
    for name, model in extractors.items():
        n = model.count_params()
        total += n
        print(f"{name:<25} {n:>10,} {model.get_feature_dim():>12}")
    print("-"*55)
    print(f"{'TOTAL':<25} {total:>10,}")
    print("="*55 + "\n")


if __name__ == "__main__":
    # Sanity check tất cả extractors
    print("=== SSVEP Feature Extractors - Sanity Check ===\n")
    
    n_channels, n_samples, n_classes = 9, 125, 40
    batch_size = 4
    
    x = torch.randn(batch_size, n_channels, n_samples)
    
    extractors = build_all_extractors(n_channels, n_samples, n_classes)
    count_params_summary(extractors)
    
    print("Forward pass check:")
    for name, model in extractors.items():
        model.eval()
        with torch.no_grad():
            logits = model(x)
            features = model.forward_features(x)
        print(f"  {name}: input={tuple(x.shape)} "
              f"→ features={tuple(features.shape)} "
              f"→ logits={tuple(logits.shape)} ✓")
    
    print("\n✓ Tất cả extractors OK")
