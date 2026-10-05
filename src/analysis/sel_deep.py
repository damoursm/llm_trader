"""Neural-net arm of the selection-objective models (user spec 2026-09-25).

Runs in the isolated torch venv (``C:\\Users\\mathi\\dlvenv`` — numpy + torch
only, see memory ``torch-isolated-venv-2026-08``), so it imports NOTHING from
the project and is run BY PATH, never ``-m``::

    C:\\Users\\mathi\\dlvenv\\Scripts\\python.exe src/analysis/sel_deep.py --side long --phase val
    C:\\Users\\mathi\\dlvenv\\Scripts\\python.exe src/analysis/sel_deep.py --side long --phase final

Per side, an MLP scores every row of the `ml30.build` arrays. The loss is the
selection objective made differentiable: within each run the scores are
standardised and turned into a sharp softmax (temperature ``--tau``), and the
loss is minus the softmax-weighted return per day of the side — the expected
return of a "soft top-1" pick (returns winsorised at the training sample's
0.5/99.5 percentiles). The output is the side's CONVICTION (higher = better
entry on that side). ``--phase val`` fits on sessions <= 2025-12-31 and
early-stops on the validation window's hard top-1 pick return; ``--phase
final`` refits on sessions <= 2026-04-30 for the epoch count the val phase
chose and scores every tradeable bar of the evaluation arrays. Predictions are
saved in `sel_models`' row order (``pred_mlp_<side>.npy``), and the project
venv scores them on the full objective (own-history rule, one entry per day,
realizable returns) with ``sel_models --score-saved``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "cache" / "ml" / "sel"
KINDS = {"intraday": (ROOT / "cache" / "ml" / "sel30", ROOT / "cache" / "ml" / "sel30_eval", ""),
         "daily": (ROOT / "cache" / "ml" / "seld", ROOT / "cache" / "ml" / "seld", "daily_")}
EVAL_FROM = "2026-05-01"                      # = sel_models.EVAL_FROM
MIN_PRICE, MIN_DV = 5.0, 5e6                  # = sel_models' Gate-4 trade floor
BARS_PER_DAY = 13
MKT_PREFIX = "dp_mkt_"                        # the market-context group (regime timing)
PHASES = {"val": dict(cut="2025-12-31", val=("2026-01-02", "2026-04-30"), tag="val"),
          "final": dict(cut="2026-04-30", val=None, tag="final")}


def dnum(iso: str) -> int:
    return int((np.datetime64(iso, "D") - np.datetime64("1970-01-01", "D")).astype(int))


def lowprio() -> None:
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)          # BELOW_NORMAL
    except Exception:                                               # noqa: BLE001
        pass


class Arrays:
    SMALL = ("y", "conf", "ba", "dn", "bar", "px", "dv20", "tk")

    def __init__(self, d: Path, fset: str):
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        base = list(self.meta["base_features"])
        deep = list(self.meta.get("deep_features") or [])
        dsel = [] if fset == "base" else [f for f in deep if fset == "all" or not f.startswith(MKT_PREFIX)]
        self.names = base + dsel
        self.dcols = [deep.index(f) for f in dsel]
        self.X = np.load(d / "X.npy", mmap_mode="r")
        self.D = np.load(d / "D.npy", mmap_mode="r") if dsel else None
        for k in self.SMALL:
            setattr(self, k, np.load(d / f"{k}.npy"))
        for k in ("xl", "xlb", "xs", "xsb"):                 # ml30.add_exit_labels, when run
            f = d / f"{k}.npy"
            setattr(self, k, np.load(f) if f.exists() else None)
        self.run = self.dn.astype(np.int64) * 100 + self.bar.astype(np.int64)

    def tradeable(self) -> np.ndarray:
        return (self.px >= MIN_PRICE) & (np.nan_to_num(self.dv20) >= MIN_DV)

    def matrix(self, rows: np.ndarray, chunk: int = 250_000) -> np.ndarray:
        nb = self.X.shape[1]
        M = np.empty((len(rows), len(self.names)), np.float32)
        for a in range(0, len(rows), chunk):
            r = rows[a:a + chunk]
            M[a:a + len(r), :nb] = self.X[r]
            if self.dcols:
                M[a:a + len(r), nb:] = self.D[r][:, self.dcols]
        return M


def side_rpd(y, ba, side, floor=1.0):
    d = np.where(ba > 0, ba, np.nan) / BARS_PER_DAY
    r = y / np.maximum(d, floor)
    return r if side == "long" else -r


TRAIL_EMBARGO_DAYS = 16                       # = sel_models.TRAIL_EMBARGO_DAYS


def side_target(arr, rows, side, target, floor=1.0):
    """= sel_models.side_target: ``pivot`` (the return per day to the next
    pivot) or ``trail`` (the return per day the pivot threshold's trailing stop
    actually captures)."""
    if target == "pivot":
        return side_rpd(arr.y[rows], arr.ba[rows], side, floor)
    if side == "long":
        r, b = arr.xl[rows].astype(float), arr.xlb[rows].astype(float)
    else:
        r, b = -arr.xs[rows].astype(float), arr.xsb[rows].astype(float)
    r = np.where(b > 0, r, np.nan)
    return r / np.maximum(b / BARS_PER_DAY, floor)


class Scaler:
    """Robust standardisation fitted on a training sample: (x - median) / IQR,
    clipped to +-5, NaN -> 0 plus a missingness column for every feature that
    is missing on more than 1% of the sample."""

    def fit(self, M: np.ndarray, max_rows: int = 400_000, seed: int = 0) -> "Scaler":
        rng = np.random.default_rng(seed)
        S = M[np.sort(rng.choice(len(M), min(len(M), max_rows), replace=False))].astype(np.float64)
        self.med = np.nan_to_num(np.nanmedian(S, axis=0))
        q1, q3 = np.nanpercentile(S, [25, 75], axis=0)
        iqr = np.nan_to_num(q3 - q1)
        sd = np.nan_to_num(np.nanstd(S, axis=0))
        self.scale = np.where(iqr > 1e-9, iqr, np.where(sd > 1e-9, sd, 1.0))
        self.mask_cols = np.flatnonzero(np.isnan(S).mean(axis=0) > 0.01)
        return self

    @property
    def n_out(self) -> int:
        return len(self.med) + len(self.mask_cols)

    def transform(self, M: np.ndarray, out: np.ndarray) -> None:
        nf = len(self.med)
        Z = (M - self.med) / self.scale
        np.clip(Z, -5.0, 5.0, out=Z)
        out[:, :nf] = np.nan_to_num(Z, nan=0.0)
        out[:, nf:] = np.isnan(M[:, self.mask_cols])


def standardized(arr: Arrays, rows: np.ndarray, scaler: Scaler, chunk: int = 250_000) -> np.ndarray:
    """The scaled matrix for ``rows`` as float16 (half the RAM; values are in
    +-5 so the precision loss is ~1e-3)."""
    Z = np.empty((len(rows), scaler.n_out), np.float16)
    buf = np.empty((min(chunk, len(rows)), scaler.n_out), np.float32)
    for a in range(0, len(rows), chunk):
        r = rows[a:a + chunk]
        scaler.transform(arr.matrix(r), buf[:len(r)])
        Z[a:a + len(r)] = buf[:len(r)]
    return Z


def build_model(n_in: int, h1: int, h2: int, dropout: float):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(n_in, h1), nn.SiLU(), nn.Dropout(dropout),
                         nn.Linear(h1, h2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(h2, 1))


def soft_top1_loss(s, r, gid, n_groups: int, tau: float):
    """Minus the mean over runs of the softmax(standardised score / tau)-
    weighted return: the expected return of a soft top-1 pick per run."""
    import torch
    cnt = torch.bincount(gid, minlength=n_groups).clamp(min=1).to(s.dtype)
    mean = torch.zeros(n_groups, dtype=s.dtype).index_add_(0, gid, s) / cnt
    dev = s - mean[gid]
    var = torch.zeros(n_groups, dtype=s.dtype).index_add_(0, gid, dev * dev) / cnt
    logit = dev / torch.sqrt(var[gid] + 1e-6) / tau
    gmax = torch.full((n_groups,), -1e9, dtype=s.dtype).scatter_reduce(0, gid, logit, reduce="amax")
    e = torch.exp(logit - gmax[gid])
    den = torch.zeros(n_groups, dtype=s.dtype).index_add_(0, gid, e)
    ret = torch.zeros(n_groups, dtype=s.dtype).index_add_(0, gid, e / den[gid] * r)
    return -ret.mean()


def predict(model, Z: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    import torch
    model.eval()
    out = np.empty(len(Z), np.float32)
    with torch.no_grad():
        for a in range(0, len(Z), chunk):
            out[a:a + chunk] = model(torch.from_numpy(Z[a:a + chunk].astype(np.float32))).squeeze(-1).numpy()
    return out


def top1_proxy(score: np.ndarray, run: np.ndarray, day: np.ndarray, r: np.ndarray, min_rows: int = 20) -> float:
    """Validation proxy: each run's highest-scored name's side return per day
    (unlabelled picks skipped), averaged per day then over days."""
    o = np.lexsort((-score, run))
    rs, first = run[o], np.r_[True, run[o][1:] != run[o][:-1]]
    starts = np.flatnonzero(first)
    sizes = np.diff(np.r_[starts, len(rs)])
    pick = o[starts[sizes >= min_rows]]
    v, d = r[pick], day[pick]
    ok = np.isfinite(v)
    if not ok.any():
        return float("nan")
    days, inv = np.unique(d[ok], return_inverse=True)
    return float((np.bincount(inv, weights=v[ok]) / np.bincount(inv)).mean())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="MLP arm of the selection-objective models")
    ap.add_argument("--side", required=True, choices=("long", "short"))
    ap.add_argument("--phase", required=True, choices=tuple(PHASES))
    ap.add_argument("--kind", default="intraday", choices=tuple(KINDS))
    ap.add_argument("--target", default="pivot", choices=("pivot", "trail"))
    ap.add_argument("--fset", default="deep", choices=("base", "deep", "all"))
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--runs-per-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--h1", type=int, default=256)
    ap.add_argument("--h2", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import torch
    lowprio()
    torch.set_num_threads(int(a.threads))
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    ph = PHASES[a.phase]
    train_dir, eval_dir, prefix = KINDS[a.kind]
    out_dir = OUT_DIR / (prefix + ph["tag"])
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"mlp_{a.side}" if a.target == "pivot" else f"mlp_{a.target}_{a.side}"
    log = open(out_dir / f"{name}.log", "a", encoding="utf-8")

    def say(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    t0 = time.time()
    arr = Arrays(train_dir, a.fset)
    c = dnum(ph["cut"])
    ok = (np.isfinite(arr.y) & (arr.ba > 0) & (arr.conf > 0) & (arr.dn <= c) & (arr.conf <= c) & arr.tradeable())
    if a.target == "trail":
        if arr.xl is None:
            raise SystemExit(f"{train_dir} has no exit labels - run ml30.add_exit_labels first")
        ok &= (np.isfinite(arr.xl) & (arr.xlb > 0) & np.isfinite(arr.xs) & (arr.xsb > 0)
               & (arr.dn <= c - TRAIL_EMBARGO_DAYS))
    tr = np.flatnonzero(ok)
    tr = tr[np.lexsort((arr.tk[tr], arr.run[tr]))]
    r_tr = side_target(arr, tr, a.side, a.target).astype(np.float64)
    lo, hi = np.nanpercentile(r_tr, [0.5, 99.5])
    r_tr = np.clip(r_tr, lo, hi).astype(np.float32)
    scaler = Scaler().fit(arr.matrix(tr[np.sort(rng.choice(len(tr), min(len(tr), 400_000), replace=False))]))
    Z = standardized(arr, tr, scaler)
    run_tr = arr.run[tr]
    starts = np.flatnonzero(np.r_[True, run_tr[1:] != run_tr[:-1]])
    ends = np.r_[starts[1:], len(tr)]
    say(f"{name} {a.phase}: {len(tr):,} rows / {len(starts):,} runs, {Z.shape[1]} inputs, "
        f"winsor [{lo:+.2f}, {hi:+.2f}] | load {time.time() - t0:.0f}s")

    val_rows = Zv = r_v = None
    if ph["val"]:
        v0, v1 = dnum(ph["val"][0]), dnum(ph["val"][1])
        val_rows = np.flatnonzero((arr.dn >= v0) & (arr.dn <= v1) & arr.tradeable())
        Zv = standardized(arr, val_rows, scaler)
        r_v = side_target(arr, val_rows, a.side, a.target)
    epochs = int(a.epochs)
    if a.phase == "final":
        vj = out_dir.parent / f"{prefix}val" / f"{name}.json"
        if vj.exists():
            epochs = int(json.loads(vj.read_text(encoding="utf-8"))["best_epoch"])
        say(f"{name} final: {epochs} epochs (from the val phase)")

    model = build_model(Z.shape[1], a.h1, a.h2, a.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    best, best_ep, best_state = -np.inf, 0, None
    K = int(a.runs_per_batch)
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(starts))
        tot, nb, te = 0.0, 0, time.time()
        for b in range(0, len(order), K):
            gs = order[b:b + K]
            idx = np.concatenate([np.arange(starts[g], ends[g]) for g in gs])
            gid = np.concatenate([np.full(ends[g] - starts[g], i) for i, g in enumerate(gs)])
            xb = torch.from_numpy(Z[idx].astype(np.float32))
            rb = torch.from_numpy(r_tr[idx])
            loss = soft_top1_loss(model(xb).squeeze(-1), rb, torch.from_numpy(gid), len(gs), a.tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()); nb += 1
        msg = f"{name} epoch {ep}: train soft-top1 {-tot / max(nb, 1):+.4f} | {time.time() - te:.0f}s"
        if Zv is not None:
            pv = predict(model, Zv)
            proxy = top1_proxy(pv, arr.run[val_rows], arr.dn[val_rows], r_v)
            msg += f" | val hard top-1 rpd {proxy:+.4f}"
            if proxy > best:
                best, best_ep = proxy, ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif ep - best_ep >= 2:
                say(msg + " | early stop")
                break
        say(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
    info = dict(kind=a.kind, target=a.target, side=a.side, phase=a.phase, fset=a.fset, tau=a.tau, lr=a.lr, h1=a.h1, h2=a.h2,
                dropout=a.dropout, runs_per_batch=K, seed=a.seed, n_train=int(len(tr)),
                n_inputs=int(Z.shape[1]), best_epoch=int(best_ep or epochs), best_val_proxy=float(best),
                cut=ph["cut"], val=ph["val"])
    del Z
    if a.phase == "val":
        np.save(out_dir / "val_rows.npy", val_rows)
        np.save(out_dir / f"pred_{name}.npy", predict(model, Zv))
    else:
        ev = Arrays(eval_dir, a.fset)
        er = np.flatnonzero(ev.tradeable() & (ev.dn >= dnum(EVAL_FROM)))
        np.save(out_dir / f"pred_{name}.npy", predict(model, standardized(ev, er, scaler)))
        np.save(out_dir / f"rows_{name}.npy", er)
    torch.save(model.state_dict(), out_dir / f"{name}.pt")
    (out_dir / f"{name}.json").write_text(json.dumps(info), encoding="utf-8")
    say(f"{name} {a.phase} done in {time.time() - t0:.0f}s: {info}")


if __name__ == "__main__":
    main()
