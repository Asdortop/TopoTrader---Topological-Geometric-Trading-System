"""
TopoTrader V3 — Dual-Branch TCN + SE-Net Channel Attention + Mixture of Experts
=================================================================================

Architecture fix for V2's core flaw:
    V2 used a flat 16-channel TCN, treating RSI (smooth daily signal) and
    H0 Persistence (rare spike signal) with identical convolutional filters.
    The model learned mostly from C1-C7 and ignored C8-C16.

V3 separates concerns:
    Branch A (TCN-A): C1-C7   — short receptive field (dilation 1-2-4-8 = 30d)
                                  Optimised for momentum/oscillator patterns
    Branch B (TCN-B): C8-C16  — long receptive field  (dilation 1-4-16-32 = 90d)
                                  Optimised for slow-building topological signals

SE-Net Channel Attention:
    After both branches, learns per-channel importance weights.
    In crash regimes: upweights Branch B (topology).
    In bull regimes:  upweights Branch A (momentum).

Mixture of Experts (MoE):
    4 small linear heads, one per regime (Crash / High-Vol / Bull / Sideways).
    Gated by the regime one-hot (C13-C16) — each expert specialises on its regime.
    Blended output = sum(gate_weight_i * expert_i(features)).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# ─────────────────────────────────────────────────────────────────────────────
# Core Building Blocks (reused from V2)
# ─────────────────────────────────────────────────────────────────────────────

class Chomp1d(nn.Module):
    """Removes right-padding to enforce causality — no future leakage."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Single dilated causal convolutional block with residual connection.
    Identical to V2 — proven stable.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 padding, dropout=0.2):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.chomp1    = Chomp1d(padding)
        self.act1      = nn.SELU()
        self.drop1     = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation))
        self.chomp2    = Chomp1d(padding)
        self.act2      = nn.SELU()
        self.drop2     = nn.Dropout(dropout)

        self.net       = nn.Sequential(self.conv1, self.chomp1, self.act1, self.drop1,
                                       self.conv2, self.chomp2, self.act2, self.drop2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


def build_tcn_branch(n_inputs, channels, kernel_size, dilations, dropout=0.2):
    """
    Build a TCN branch with custom dilation schedule.

    Branch A uses dilations [1,2,4,8]   → receptive field = 30 days
    Branch B uses dilations [1,4,16,32] → receptive field = 90 days
    """
    layers = []
    for idx, dilation in enumerate(dilations):
        in_ch  = n_inputs if idx == 0 else channels[idx - 1]
        out_ch = channels[idx]
        pad    = (kernel_size - 1) * dilation
        layers.append(TemporalBlock(in_ch, out_ch, kernel_size,
                                    stride=1, dilation=dilation,
                                    padding=pad, dropout=dropout))
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# SE-Net Channel Attention
# ─────────────────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention block.

    After both TCN branches produce feature maps, SEBlock learns a weight
    vector w ∈ [0,1]^C — one scalar per channel — via global average pooling
    followed by a 2-layer MLP bottleneck.

    During crash: w[topological_channels] → 0.8, w[momentum_channels] → 0.2
    During bull:  reversed — model learns this from data, not hardcoded.

    Parameters
    ----------
    n_channels : int — total channels after branch concatenation (2 × branch_ch)
    reduction  : int — bottleneck factor for the MLP (default 4)
    """
    def __init__(self, n_channels, reduction=4):
        super().__init__()
        bottleneck = max(n_channels // reduction, 4)
        self.gap   = nn.AdaptiveAvgPool1d(1)             # global context per channel
        self.fc1   = nn.Linear(n_channels, bottleneck)
        self.fc2   = nn.Linear(bottleneck, n_channels)

    def forward(self, x):
        # x: (B, C, T)
        w = self.gap(x).squeeze(-1)                      # (B, C)
        w = F.relu(self.fc1(w))                          # (B, bottleneck)
        w = torch.sigmoid(self.fc2(w)).unsqueeze(-1)     # (B, C, 1)
        return x * w                                     # channel-wise rescaling


# ─────────────────────────────────────────────────────────────────────────────
# Mixture of Experts Head
# ─────────────────────────────────────────────────────────────────────────────

class MoEHead(nn.Module):
    """
    Mixture of Experts prediction head.

    4 lightweight expert linear heads — one per market regime:
        Expert 0: Crash      (C13=1)
        Expert 1: High-Vol   (C14=1)
        Expert 2: Bull       (C15=1)
        Expert 3: Sideways   (C16=1)

    Gate reads the regime one-hot from C13-C16 and produces a soft blend.
    Each expert only learns from its own regime's examples (effectively),
    because the gate assigns near-zero weight to wrong-regime experts.

    Parameters
    ----------
    in_features  : int — size of the fused feature vector (from SE-Net output)
    n_experts    : int — number of regime experts (4)
    n_regime_ch  : int — number of regime channels in input (4, for C13-C16)
    """
    def __init__(self, in_features, n_experts=4, n_regime_ch=4):
        super().__init__()
        self.n_experts = n_experts

        # Expert heads — each is a 2-layer MLP for richer specialisation
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, 16),
                nn.SELU(),
                nn.Linear(16, 1)
            )
            for _ in range(n_experts)
        ])

        # Gate: reads regime one-hot → soft mixture weights
        self.gate = nn.Linear(n_regime_ch, n_experts)

    def forward(self, features, regime_onehot):
        """
        Parameters
        ----------
        features      : (B, in_features) — fused representation from SE-Net
        regime_onehot : (B, 4)           — C13-C16 one-hot regime labels

        Returns
        -------
        (B, 1) — blended prediction probability (before sigmoid)
        """
        gate_weights = F.softmax(self.gate(regime_onehot), dim=-1)   # (B, 4)

        expert_outs = torch.stack(
            [self.experts[i](features) for i in range(self.n_experts)],
            dim=1
        )                                                              # (B, 4, 1)

        blended = (expert_outs * gate_weights.unsqueeze(-1)).sum(dim=1)  # (B, 1)
        return blended


# ─────────────────────────────────────────────────────────────────────────────
# V3 Full Model
# ─────────────────────────────────────────────────────────────────────────────

class MarketTCN_V3(nn.Module):
    """
    TopoTrader V3 — Dual-Branch TCN + SE-Net + Mixture of Experts.

    Channel layout (matches V2 feature engineering):
        C1-C7   : Standard technical (LogRet, Vol, RSI, MACD, ATR, BB, ZScore)
        C8-C12  : Geometric/Topological (GAT, Walsh, H0, H1, Beta)
        C13-C16 : Regime one-hot (Crash, High-Vol, Bull, Sideways)

    Branch A handles C1-C7   → short dilation [1,2,4,8],   RF=30d
    Branch B handles C8-C16  → long dilation  [1,4,16,32], RF=90d

    Parameters
    ----------
    n_tech_ch    : technical channels (default 7, C1-C7)
    n_topo_ch    : topological + regime channels (default 9, C8-C16)
    branch_ch    : hidden channels per TemporalBlock in each branch
    kernel_size  : convolution kernel (default 3)
    dropout      : dropout probability (default 0.2)
    se_reduction : SE-Net bottleneck reduction factor (default 4)
    n_experts    : number of MoE experts (default 4, one per regime)
    n_regime_ch  : regime channels fed to MoE gate (default 4, C13-C16)
    """
    def __init__(
        self,
        n_tech_ch   = 7,
        n_topo_ch   = 9,
        branch_ch   = [32, 32, 32, 32],
        kernel_size = 3,
        dropout     = 0.2,
        se_reduction= 4,
        n_experts   = 4,
        n_regime_ch = 4,
    ):
        super().__init__()

        self.n_tech_ch   = n_tech_ch
        self.n_topo_ch   = n_topo_ch
        self.n_regime_ch = n_regime_ch

        # ── Branch A: short dilation for momentum signals ──────────────────
        self.branch_a = build_tcn_branch(
            n_inputs   = n_tech_ch,
            channels   = branch_ch,
            kernel_size= kernel_size,
            dilations  = [1, 2, 4, 8],     # RF = 30 days
            dropout    = dropout,
        )

        # ── Branch B: long dilation for topological signals ────────────────
        self.branch_b = build_tcn_branch(
            n_inputs   = n_topo_ch,
            channels   = branch_ch,
            kernel_size= kernel_size,
            dilations  = [1, 4, 16, 32],   # RF = 90 days
            dropout    = dropout,
        )

        # After concat: 2 × branch_ch[-1] channels
        fused_ch = 2 * branch_ch[-1]

        # ── SE-Net channel attention ───────────────────────────────────────
        self.se = SEBlock(n_channels=fused_ch, reduction=se_reduction)

        # ── Fusion projection: (B, fused_ch, T) → (B, fused_ch) ──────────
        self.gap_fused = nn.AdaptiveAvgPool1d(1)

        # ── Mixture of Experts head ────────────────────────────────────────
        self.moe = MoEHead(
            in_features  = fused_ch,
            n_experts    = n_experts,
            n_regime_ch  = n_regime_ch,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Parameters
        ----------
        x : (B, 16, T) — full 16-channel input tensor

        Returns
        -------
        (B, 1) — P(next-day return > 0)
        """
        # Split channels
        x_tech   = x[:, :self.n_tech_ch, :]                     # (B, 7, T)
        x_topo   = x[:, self.n_tech_ch:, :]                     # (B, 9, T) — C8-C16
        # Regime one-hot: last 4 channels of x_topo, at last time step
        regime   = x[:, -self.n_regime_ch:, -1]                 # (B, 4)

        # Branch A: momentum features
        a_out    = self.branch_a(x_tech)                         # (B, 32, T)

        # Branch B: topological + regime features
        b_out    = self.branch_b(x_topo)                         # (B, 32, T)

        # Concat along channel dim
        fused    = torch.cat([a_out, b_out], dim=1)              # (B, 64, T)

        # SE-Net: learn per-channel importance
        fused    = self.se(fused)                                 # (B, 64, T)

        # Global average pool → fixed-size feature vector
        fused_v  = self.gap_fused(fused).squeeze(-1)             # (B, 64)

        # MoE: regime-gated blended prediction
        logit    = self.moe(fused_v, regime)                     # (B, 1)

        return self.sigmoid(logit)                               # (B, 1) ∈ [0,1]


# ─────────────────────────────────────────────────────────────────────────────
# Magnitude-Weighted BCE Loss
# ─────────────────────────────────────────────────────────────────────────────

class MagnitudeWeightedBCE(nn.Module):
    """
    BCE loss weighted by the magnitude of the actual next-day return.

    Rationale: a correct prediction on a 3% move day should count more
    than on a 0.1% move day — both for model learning and for real PnL.

    Weight scaling: w = 0.5 + 1.5 × (|return| / max(|return|))
        → small moves get 0.5× weight (still contribute to gradient)
        → large moves get 2.0× weight (4× more influence than small moves)

    Parameters
    ----------
    magnitudes : (N,) absolute next-day returns, passed per batch
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, magnitudes=None):
        """
        Parameters
        ----------
        pred       : (B, 1) model output probabilities
        target     : (B, 1) binary labels
        magnitudes : (B,) absolute returns — None falls back to plain BCE
        """
        bce = F.binary_cross_entropy(pred, target, reduction='none')  # (B, 1)

        if magnitudes is None:
            return bce.mean()

        max_mag = magnitudes.max().clamp(min=1e-8)
        w = (0.5 + 1.5 * (magnitudes / max_mag)).unsqueeze(1)         # (B, 1)
        return (bce * w).mean()
