"""
06_published_arch_diagnostic.py - training-length sensitivity diagnostic.

Trains the architectures listed in Supplementary Table S2 on a 78/20 split and records the
TEST R2 after 10 / 50 / 100 / 200 / 400 / 800 epochs, for full-batch and batch-size-32
training, several seeds.
    LSTM : 3 layers x 256, dropout 0.3, Adam lr 1e-3, weight decay 1e-4   (1,325,313 parameters, per-time-point input)
    MLP  : [128, 64, 16] ReLU, dropout 0.3, Adam lr 1e-2, weight decay 1e-3
usage: python src/06_published_arch_diagnostic.py [--strategy C0] [--seeds 3] [--epochs 800]
"""
import argparse, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from cv_pipeline import prepare_fold, _zc
import models as M
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser(); ap.add_argument("--strategy", default="C0"); ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--epochs", type=int, default=800); ap.add_argument("--data", default=os.path.join(DATA, "patients.csv")); a = ap.parse_args()
df = load_patients(a.data); y = auc24_reference(df)
tr, te = train_test_split(np.arange(len(df)), test_size=20, random_state=42)
F = prepare_fold(df, tr, te, seed=42)
concs = STRATEGIES[a.strategy]
zc_tr, zc_te, zk_tr, zk_te = _zc(F, COV7, "tr"), _zc(F, COV7, "te"), _zc(F, concs, "tr"), _zc(F, concs, "te")
mu, sd = y[tr].mean(), y[tr].std(); ys = ((y[tr] - mu) / sd).astype(np.float32)
CK = [10, 50, 100, 200, 400, 800]; CK = [c for c in CK if c <= a.epochs]

class MLPpub(nn.Module):
    def __init__(s, d):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(.3), nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(s, x): return s.net(x).squeeze(-1)

rows = []
for kind in ("lstm", "mlp"):
    Xtr = torch.as_tensor(M.build_inputs(kind, zc_tr, zk_tr)); Xte = torch.as_tensor(M.build_inputs(kind, zc_te, zk_te)); yt = torch.as_tensor(ys)
    for bs in (None, 32):
        for seed in range(a.seeds):
            torch.manual_seed(seed)
            net = M.SeqLSTM(8, 256, layers=3, dropout=0.3) if kind == "lstm" else MLPpub(Xtr.shape[1])
            opt = torch.optim.Adam(net.parameters(), lr=1e-3 if kind == "lstm" else 1e-2, weight_decay=1e-4 if kind == "lstm" else 1e-3)
            t0 = time.time()
            for ep in range(1, a.epochs + 1):
                net.train()
                if bs is None:
                    opt.zero_grad(); nn.functional.mse_loss(net(Xtr), yt).backward(); opt.step()
                else:
                    perm = torch.randperm(len(yt))
                    for s in range(0, len(yt), bs):
                        b = perm[s:s + bs]; opt.zero_grad(); nn.functional.mse_loss(net(Xtr[b]), yt[b]).backward(); opt.step()
                if ep in CK:
                    net.eval()
                    with torch.no_grad(): p = net(Xte).numpy() * sd + mu
                    rows.append(dict(model=kind, batch=("full" if bs is None else bs), seed=seed, epochs=ep, test_R2=metrics(y[te], p)["R2"]))
            print(kind, "batch", bs, "seed", seed, f"{time.time()-t0:.0f}s", flush=True)
out = pd.DataFrame(rows); out.to_csv(os.path.join(RES, "published_arch_diagnostic_raw.csv"), index=False)
piv = out.groupby(["model", "batch", "epochs"]).test_R2.agg(["mean", "std"]).round(3).reset_index()
piv.to_csv(os.path.join(RES, "published_arch_diagnostic.csv"), index=False); print(piv.to_string())
