"""
TopoTrader V3 — Ablation Study (Week 7)
========================================
Trains 4 ablated V3 variants to prove each component earns its place.
Compares pooled hit rate against full V3 baseline.

Ablations:
  1. No SE-Net       — flat concat instead of channel attention
  2. No MoE          — single linear head instead of regime-expert heads
  3. No Dual-Branch  — flat 16-ch TCN (effectively V2 architecture)
  4. Plain BCE       — no magnitude weighting (same architecture, different loss)

Usage:
    python run_ablation_v3.py

Output:
    reports/ablation_v3.csv  — per-ablation pooled hit rates
    Printed contribution table
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topo_trader.models.tcn import MarketTCN
from topo_trader.models.tcn_v3 import (
    MarketTCN_V3, MagnitudeWeightedBCE,
    Chomp1d, TemporalBlock, build_tcn_branch, SEBlock,
)
from topo_trader.train import train_model_v3
from topo_trader.evaluation.walk_forward import (
    INDIA_WALK_FORWARD_WINDOWS, create_dataset_for_range,
)
from topo_trader.utils.data_loader import (
    fetch_india_csv_tickers, load_india_csv_data, generate_features,
)
from statsmodels.stats.proportion import proportion_confint

WINDOW_LEN  = 64
EPOCHS      = 30
LR          = 1e-3
LONG_THRESH = 0.55
SHORT_THRESH= 0.45
BREAK_EVEN  = 0.5010

os.makedirs("reports", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Ablated Model Variants
# ─────────────────────────────────────────────────────────────────────────────

class MarketTCN_V3_NoSE(nn.Module):
    """V3 without SE-Net — flat concat, no channel attention."""
    def __init__(self, n_tech_ch=7, n_topo_ch=9, branch_ch=[32,32,32,32],
                 kernel_size=3, dropout=0.2, n_experts=4, n_regime_ch=4):
        super().__init__()
        self.n_tech_ch   = n_tech_ch
        self.n_topo_ch   = n_topo_ch
        self.n_regime_ch = n_regime_ch
        self.branch_a = build_tcn_branch(n_tech_ch, branch_ch, kernel_size,
                                         [1,2,4,8], dropout)
        self.branch_b = build_tcn_branch(n_topo_ch, branch_ch, kernel_size,
                                         [1,4,16,32], dropout)
        fused_ch = 2 * branch_ch[-1]
        self.gap_fused = nn.AdaptiveAvgPool1d(1)
        # MoE head
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(fused_ch, 16), nn.SELU(), nn.Linear(16, 1))
            for _ in range(n_experts)
        ])
        self.gate    = nn.Linear(n_regime_ch, n_experts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_tech  = x[:, :self.n_tech_ch, :]
        x_topo  = x[:, self.n_tech_ch:, :]
        regime  = x[:, -self.n_regime_ch:, -1]
        a_out   = self.branch_a(x_tech)
        b_out   = self.branch_b(x_topo)
        fused   = torch.cat([a_out, b_out], dim=1)        # no SE-Net here
        fused_v = self.gap_fused(fused).squeeze(-1)
        gates   = F.softmax(self.gate(regime), dim=-1)
        expert_outs = torch.stack([e(fused_v) for e in self.experts], dim=1)
        out = (expert_outs * gates.unsqueeze(-1)).sum(dim=1)
        return self.sigmoid(out)


class MarketTCN_V3_NoMoE(nn.Module):
    """V3 without MoE — single linear head, no regime specialisation."""
    def __init__(self, n_tech_ch=7, n_topo_ch=9, branch_ch=[32,32,32,32],
                 kernel_size=3, dropout=0.2, se_reduction=4):
        super().__init__()
        self.n_tech_ch   = n_tech_ch
        self.n_topo_ch   = n_topo_ch
        self.branch_a = build_tcn_branch(n_tech_ch, branch_ch, kernel_size,
                                         [1,2,4,8], dropout)
        self.branch_b = build_tcn_branch(n_topo_ch, branch_ch, kernel_size,
                                         [1,4,16,32], dropout)
        fused_ch = 2 * branch_ch[-1]
        self.se         = SEBlock(n_channels=fused_ch, reduction=se_reduction)
        self.gap_fused  = nn.AdaptiveAvgPool1d(1)
        self.head       = nn.Sequential(
            nn.Linear(fused_ch, 16), nn.SELU(), nn.Linear(16, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_tech  = x[:, :self.n_tech_ch, :]
        x_topo  = x[:, self.n_tech_ch:, :]
        a_out   = self.branch_a(x_tech)
        b_out   = self.branch_b(x_topo)
        fused   = self.se(torch.cat([a_out, b_out], dim=1))
        fused_v = self.gap_fused(fused).squeeze(-1)
        return self.sigmoid(self.head(fused_v))


# ─────────────────────────────────────────────────────────────────────────────
# Generic trainer for ablated models
# ─────────────────────────────────────────────────────────────────────────────

def train_ablation(model, X_train, y_train, magnitudes=None,
                   use_mag_loss=True, epochs=EPOCHS, lr=LR, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = MagnitudeWeightedBCE() if use_mag_loss else nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr/100)
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    mag_t = torch.tensor(magnitudes, dtype=torch.float32).to(device) \
            if (magnitudes is not None and use_mag_loss) else None
    best, patience, wait = float('inf'), 10, 0
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X_train))
        epoch_loss, n = 0.0, 0
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i+batch_size]
            bx, by = X_t[idx], y_t[idx]
            bmag = mag_t[idx] if mag_t is not None else None
            opt.zero_grad()
            if use_mag_loss and bmag is not None:
                loss = criterion(model(bx), by, bmag)
            else:
                loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item(); n += 1
        sch.step()
        mean_loss = epoch_loss / max(n, 1)
        if mean_loss < best - 1e-5:
            best = mean_loss; wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.cpu()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def pooled_hit_rate(model, X_test, y_test):
    model.eval()
    model.cpu()   # ensure on CPU regardless of where training ran
    with torch.no_grad():
        probs = model(torch.tensor(X_test, dtype=torch.float32)).flatten().numpy()
    conf  = (probs > LONG_THRESH) | (probs < SHORT_THRESH)
    if conf.sum() == 0:
        return 0.5, 0.0
    preds = (probs[conf] > 0.5).astype(int)
    hr    = (preds == y_test[conf].astype(int)).mean()
    return float(hr), float(conf.mean())


def get_magnitudes(ticker_features, tickers, start, end, wlen):
    mags = []
    for t in tickers:
        if t not in ticker_features: continue
        df = ticker_features[t]
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        dr = df.loc[mask]
        if len(dr) < wlen + 1 or 'C1_LogRet' not in dr.columns: continue
        lr = dr['C1_LogRet'].values
        for i in range(len(lr) - wlen - 1):
            mags.append(abs(float(lr[i + wlen])))
    return np.array(mags, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main ablation runner
# ─────────────────────────────────────────────────────────────────────────────

def load_features():
    tickers = fetch_india_csv_tickers()
    data = load_india_csv_data(start_date="2010-01-01", end_date="2021-04-30",
                               cache_name="india_csv_data.parquet")
    tickers = list(data.columns.get_level_values(0).unique())
    feats, _ = generate_features(data, tickers, parallel=False, n_jobs=1)
    return feats, tickers


ABLATIONS = {
    "Full V3":        lambda: MarketTCN_V3(),
    "No SE-Net":      lambda: MarketTCN_V3_NoSE(),
    "No MoE":         lambda: MarketTCN_V3_NoMoE(),
    "No Dual-Branch": lambda: MarketTCN(num_inputs=16, num_channels=[32,32,32,32]),
    "Plain BCE":      lambda: MarketTCN_V3(),   # same arch, different loss
}

USE_MAG = {
    "Full V3": True, "No SE-Net": True, "No MoE": True,
    "No Dual-Branch": False, "Plain BCE": False,
}


def run():
    print("Loading features ...", flush=True)
    feats, tickers = load_features()
    windows = INDIA_WALK_FORWARD_WINDOWS[2:7]

    # Collect per-window predictions for each ablation
    all_hits   = {k: [] for k in ABLATIONS}
    all_confs  = {k: [] for k in ABLATIONS}
    all_n      = {k: [] for k in ABLATIONS}

    for i, (ts, te, vs, ve, label) in enumerate(windows):
        print(f"\n[Window {i+1}/5] {label}", flush=True)
        X_tr, y_tr, stats = create_dataset_for_range(feats, tickers, WINDOW_LEN, ts, te)
        X_te, y_te, _     = create_dataset_for_range(feats, tickers, WINDOW_LEN, vs, ve,
                                                      channel_stats=stats)
        if len(X_tr) == 0 or len(X_te) == 0: continue
        mags = get_magnitudes(feats, tickers, ts, te, WINDOW_LEN)
        min_n = min(len(X_tr), len(mags))
        X_tr2, y_tr2, mags2 = X_tr[:min_n], y_tr[:min_n], mags[:min_n]

        for name, make_fn in ABLATIONS.items():
            print(f"  [{name}] training ...", flush=True)
            model = make_fn()
            if name == "No Dual-Branch":
                # flat TCN needs plain BCE and no magnitude loss
                from topo_trader.train import train_model
                model = train_model(X_tr, y_tr, epochs=EPOCHS, lr=LR)
            else:
                model = train_ablation(model, X_tr2, y_tr2, mags2,
                                       use_mag_loss=USE_MAG[name])
            hr, cov = pooled_hit_rate(model, X_te, y_te)
            all_hits[name].append(hr)
            all_confs[name].append(cov)
            all_n[name].append(len(X_te))
            print(f"    HR={hr:.4f}  Coverage={cov:.1%}", flush=True)

    # ── Summary ────────────────────────────────────────────────────────────
    from scipy import stats as sp
    print(f"\n{'='*65}")
    print("ABLATION STUDY — CONTRIBUTION TABLE")
    print(f"{'='*65}")
    print(f"{'Variant':<20} {'Mean HR':>9} {'Cov':>7} {'vs Full V3':>12} {'p-val':>8}")
    print(f"{'-'*65}")

    rows = []
    full_hrs = np.array(all_hits["Full V3"])

    for name in ABLATIONS:
        hrs  = np.array(all_hits[name])
        covs = np.array(all_confs[name])
        mean_hr  = hrs.mean()
        mean_cov = covs.mean()
        gap = (mean_hr - full_hrs.mean()) * 100
        if name != "Full V3" and len(hrs) >= 3:
            _, p = sp.ttest_rel(hrs, full_hrs)
        else:
            p = 1.0
        flag = "BETTER" if gap > 0 else ("WORSE" if gap < -0.05 else "~EQUAL")
        print(f"  {name:<18} {mean_hr:>9.4f} {mean_cov:>7.1%} {gap:>+11.2f}pp {p:>8.4f}  {flag}")
        rows.append({"variant": name, "mean_hr": mean_hr, "mean_cov": mean_cov,
                     "gap_vs_full": gap, "p_value": round(p, 4)})

    print(f"\nInterpretation:")
    print("  If ablation HR < Full V3 -> that component is helping")
    print("  If ablation HR >= Full V3 -> that component may not be adding value")

    pd.DataFrame(rows).to_csv("reports/ablation_v3.csv", index=False)
    print("\nSaved -> reports/ablation_v3.csv")


if __name__ == "__main__":
    import pandas as pd
    run()
