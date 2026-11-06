"""
models.py - learners used in the analysis.

Per-time-point LSTM ("SeqLSTM")
-------------------------------
For a strategy that uses k concentrations the patient is a SEQUENCE of length k (time-ordered).
Step j input  = [ 7 standardised covariates (same at every step) , standardised concentration at the j-th time ]
              -> d_step = 8 (matches the 585 / 1,325,313 parameter counts of Supplementary Table S3).
The hidden state is carried from step to step; the LAST hidden state feeds a linear head -> AUC0-24.
k = 1 (the C0 strategy) gives a length-1 sequence: the recurrent weights are never used (h0 = c0 = 0)
and the network is functionally a gated feed-forward net (the recurrent weights then receive no gradient).

Capacity-matched feed-forward comparator ("FFNet")
--------------------------------------------------
Same inputs, flattened to [7 covariates, k concentrations]; two hidden ReLU layers whose width is
chosen so that the number of trainable parameters is as close as possible to the LSTM's.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)


class SeqLSTM(nn.Module):
    def __init__(self, d_step: int, hidden: int, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(d_step, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):                       # x: (n, T, d_step)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class FFNet(nn.Module):
    def __init__(self, d_in: int, width: int, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = width
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def lstm_param_count(d_step: int, hidden: int, layers: int = 1) -> int:
    tot = 4 * hidden * (d_step + hidden + 2)
    for _ in range(layers - 1):
        tot += 4 * hidden * (hidden + hidden + 2)
    return tot + hidden + 1


def match_width(target: int, d_in: int, depth: int = 2) -> int:
    """Hidden width of a depth-2 FFNet whose parameter count is closest to `target`."""
    best_w, best_gap = 2, 10 ** 12
    for w in range(2, 600):
        p = d_in * w + w + (depth - 1) * (w * w + w) + w + 1
        if abs(p - target) < best_gap:
            best_w, best_gap = w, abs(p - target)
    return best_w


def build_inputs(kind: str, zcov: np.ndarray, zconc: np.ndarray) -> np.ndarray:
    """zcov (n, c) standardised covariates (c may be 0 for the covariate-free variants);
    zconc (n, k) standardised concentrations in time order.
    kind 'lstm' -> (n, k, c+1) sequence;  kind 'mlp' -> (n, c+k) flat vector."""
    n, k = zconc.shape
    if kind == "lstm":
        steps = [np.concatenate([zcov, zconc[:, j:j + 1]], axis=1) for j in range(k)]
        return np.stack(steps, axis=1).astype(np.float32)
    return np.concatenate([zcov, zconc], axis=1).astype(np.float32)


def _train(model, Xtr, ytr, Xva=None, yva=None, *, lr=0.01, wd=1e-3, max_epochs=300, patience=30,
           min_epochs=20, fixed_epochs=None, batch_size=None, seed=0, clip=5.0):
    """Adam + MSE. Full batch unless batch_size is given. With a validation set: early stopping
    (best-epoch weights restored). With fixed_epochs: exactly that many epochs, no validation."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    xt = torch.as_tensor(Xtr); yt = torch.as_tensor(ytr, dtype=torch.float32)
    n = len(yt)
    use_val = Xva is not None and fixed_epochs is None
    if use_val:
        xv = torch.as_tensor(Xva); yv = torch.as_tensor(yva, dtype=torch.float32)
    best, best_state, best_ep, bad = float("inf"), None, 0, 0
    n_ep = fixed_epochs if fixed_epochs is not None else max_epochs
    for ep in range(1, n_ep + 1):
        model.train()
        if batch_size is None or batch_size >= n:
            opt.zero_grad(); loss = torch.mean((model(xt) - yt) ** 2); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip); opt.step()
        else:
            perm = torch.randperm(n, generator=g)
            for s in range(0, n, batch_size):
                b = perm[s:s + batch_size]
                opt.zero_grad(); loss = torch.mean((model(xt[b]) - yt[b]) ** 2); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip); opt.step()
        if use_val:
            model.eval()
            with torch.no_grad():
                v = float(torch.mean((model(xv) - yv) ** 2))
            if v < best - 1e-6:
                best, best_state, best_ep, bad = v, copy.deepcopy(model.state_dict()), ep, 0
            else:
                bad += 1
                if ep >= min_epochs and bad >= patience:
                    break
    if use_val and best_state is not None:
        model.load_state_dict(best_state)
    return model, (best_ep if use_val else n_ep), best


def _predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X)).numpy()


NN_CONFIGS = [dict(hidden=16, wd=1e-3), dict(hidden=32, wd=1e-3), dict(hidden=64, wd=1e-3), dict(hidden=32, wd=1e-2)]


def fit_predict_nn(kind, zcov_tr, zconc_tr, y_tr, zcov_te, zconc_te, val_mask, seed,
                   configs=NN_CONFIGS, lr=0.01, max_epochs=300, patience=30, return_model=False):
    """Tune over `configs` on an inner validation split (val_mask, boolean over the training rows),
    refit the winner on the whole training fold for the best number of epochs, predict the test fold.
    The MLP width is matched to the LSTM parameter count of the same config (capacity matching)."""
    mu, sd = y_tr.mean(), y_tr.std() + 1e-9
    ys = (y_tr - mu) / sd
    Xtr = build_inputs(kind, zcov_tr, zconc_tr)
    Xte = build_inputs(kind, zcov_te, zconc_te)
    d_step = zcov_tr.shape[1] + 1
    k = zconc_tr.shape[1]
    d_flat = zcov_tr.shape[1] + k

    def make(cfg):
        if kind == "lstm":
            return SeqLSTM(d_step, cfg["hidden"], layers=1)
        w = match_width(lstm_param_count(d_step, cfg["hidden"]), d_flat)
        return FFNet(d_flat, w)

    best = None
    for ci, cfg in enumerate(configs):
        torch.manual_seed(seed * 7 + ci)
        m = make(cfg)
        m, ep, vloss = _train(m, Xtr[~val_mask], ys[~val_mask], Xtr[val_mask], ys[val_mask], lr=lr, wd=cfg["wd"],
                              max_epochs=max_epochs, patience=patience, seed=seed)
        if best is None or vloss < best[0]:
            best = (vloss, ci, ep)
    _, ci, ep = best
    cfg = configs[ci]
    torch.manual_seed(seed * 7 + ci)
    m = make(cfg)
    m, _, _ = _train(m, Xtr, ys, fixed_epochs=max(ep, 20), lr=lr, wd=cfg["wd"], seed=seed)
    pred = _predict(m, Xte) * sd + mu
    info = dict(config=ci, hidden=cfg["hidden"], epochs=int(ep), params=n_params(m))
    if return_model:
        return pred, info, (m, mu, sd)
    return pred, info


def predict_with(model_pack, kind, zcov, zconc):
    m, mu, sd = model_pack
    return _predict(m, build_inputs(kind, zcov, zconc)) * sd + mu


# --------------------------------------------------------------------------- classical learners
def ols_fit_predict(Xtr, ytr, Xte):
    A = np.c_[np.ones(len(Xtr)), Xtr]
    beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return np.c_[np.ones(len(Xte)), Xte] @ beta


def loglin_fit_predict(Xtr_log, ytr, Xte_log):
    """log(AUC) ~ log-predictors, back-transformed with Duan's smearing factor."""
    A = np.c_[np.ones(len(Xtr_log)), Xtr_log]
    beta, *_ = np.linalg.lstsq(A, np.log(ytr), rcond=None)
    smear = np.mean(np.exp(np.log(ytr) - A @ beta))
    return np.exp(np.c_[np.ones(len(Xte_log)), Xte_log] @ beta) * smear


def tree_fit_predict(name, Xtr, ytr, Xte, val_mask, seed):
    """RF / XGB with the same budget as the networks: 4 configs, chosen on the inner validation split,
    winner refit on the whole training fold."""
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    if name == "RF":
        cfgs = [dict(max_depth=d) for d in (2, 3, 5, None)]
        mk = lambda c: RandomForestRegressor(n_estimators=200, max_depth=c["max_depth"], min_samples_leaf=2,
                                             random_state=seed, n_jobs=1)
    else:
        cfgs = [dict(max_depth=1, learning_rate=0.05), dict(max_depth=2, learning_rate=0.05),
                dict(max_depth=3, learning_rate=0.05), dict(max_depth=4, learning_rate=0.03)]
        mk = lambda c: XGBRegressor(n_estimators=200, max_depth=c["max_depth"], learning_rate=c["learning_rate"],
                                    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, n_jobs=1,
                                    random_state=seed, verbosity=0, tree_method="hist")
    best = (float("inf"), 0)
    for ci, c in enumerate(cfgs):
        m = mk(c).fit(Xtr[~val_mask], ytr[~val_mask])
        v = float(np.mean((m.predict(Xtr[val_mask]) - ytr[val_mask]) ** 2))
        if v < best[0]:
            best = (v, ci)
    m = mk(cfgs[best[1]]).fit(Xtr, ytr)
    return m.predict(Xte)
