"""
Blind matching between two embedding spaces via Gromov-Wasserstein.
 
Input: two .npz files, each with
    X      (N, D)  embeddings for the SAME N clips/samples, in the same order
    labels (N,)    concept label per sample (used only as ground truth)
 
Levels:
    class   average embeddings per class -> match C centroids to C centroids
    sample  match N individual samples (subsampled to --n)
 
Usage:
    python blind_match.py --a dino.npz --b beats.npz --level class
    python blind_match.py --a dino.npz --b beats.npz --level sample --n 1000
    python blind_match.py --synthetic            # pipeline sanity check
"""
import argparse, json
import numpy as np
import ot
from scipy.optimize import linear_sum_assignment
 
 
# ---------- preprocessing (anisotropy countermeasures) ----------
def preprocess(X, dim=64, standardize=True, whiten=False):
    """Center, per-dimension standardize, PCA to a common dim, optional whiten."""
    X = X.astype(np.float64)
    X = X - X.mean(0)
    if standardize:
        X = X / (X.std(0) + 1e-8)
    k = min(dim, X.shape[0] - 1, X.shape[1])
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Z = U[:, :k] * S[:k]
    if whiten:
        Z = Z / (S[:k] / np.sqrt(X.shape[0] - 1) + 1e-8)
    return Z
 
 
def dist_matrix(Z):
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    C = 1.0 - Z @ Z.T
    np.fill_diagonal(C, 0.0)
    C = np.clip(C, 0, None)
    return C / (C.max() + 1e-12)
 
 
# ---------- data ----------
def class_centroids(X, labels, min_count=3):
    classes = sorted(set(labels.tolist()))
    keep, cents = [], []
    for c in classes:
        m = labels == c
        if m.sum() >= min_count:
            keep.append(c)
            cents.append(X[m].mean(0))
    return np.stack(cents), keep
 
 
def load(path):
    d = np.load(path, allow_pickle=True)
    return d["X"], d["labels"]
 
 
# ---------- Gromov-Wasserstein with restarts ----------
def random_coupling(n, m, rng):
    K = rng.random((n, m)) + 1e-3
    p, q = np.ones(n) / n, np.ones(m) / m
    return ot.sinkhorn(p, q, -np.log(K), reg=1.0)
 
 
def gw_blind_match(C1, C2, n_restarts=10, entropic_eps=None, seed=0, max_iter=200):
    n, m = C1.shape[0], C2.shape[0]
    p, q = np.ones(n) / n, np.ones(m) / m
    rng = np.random.default_rng(seed)
    best_plan, best_val, vals = None, np.inf, []
    for r in range(n_restarts):
        G0 = None if r == 0 else random_coupling(n, m, rng)  # r=0: default init
        if entropic_eps is None:
            plan, log = ot.gromov.gromov_wasserstein(
                C1, C2, p, q, "square_loss", G0=G0, log=True, max_iter=max_iter)
        else:
            plan, log = ot.gromov.entropic_gromov_wasserstein(
                C1, C2, p, q, "square_loss", epsilon=entropic_eps, G0=G0, log=True)
        val = float(log["gw_dist"])
        vals.append(val)
        if val < best_val:
            best_val, best_plan = val, plan
    return best_plan, best_val, vals
 
 
# ---------- metrics ----------
def retrieval_metrics(plan, ks=(1, 5, 10)):
    """Ground truth = identity (row i <-> col i)."""
    n = plan.shape[0]
    ranks = (-plan).argsort(1).argsort(1)[np.arange(n), np.arange(n)]
    out = {f"top{k}": float((ranks < k).mean()) for k in ks if k <= plan.shape[1]}
    r, c = linear_sum_assignment(-plan)  # Hungarian on the plan
    out["hungarian_top1"] = float((r == c).mean())
    out["chance_top1"] = 1.0 / n
    return out
 
 
def mutual_knn(Za, Zb, k=10):
    """Mean overlap of k-NN sets (cosine) between two spaces, same samples."""
    def knn(Z):
        Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
        S = Zn @ Zn.T
        np.fill_diagonal(S, -np.inf)
        return np.argsort(-S, axis=1)[:, :k]
    A, B = knn(Za), knn(Zb)
    return float(np.mean([len(set(A[i]) & set(B[i])) / k for i in range(len(A))]))
 
 
def permutation_null(C1, C2, n_perm=20, seed=0, **kw):
    """GW top-1 when ground truth is destroyed (shuffle one space)."""
    rng = np.random.default_rng(seed)
    n = C1.shape[0]
    res = []
    for _ in range(n_perm):
        perm = rng.permutation(n)
        plan, _, _ = gw_blind_match(C1, C2[np.ix_(perm, perm)], n_restarts=1,
                                    seed=int(rng.integers(1 << 30)), **kw)
        # truth under shuffle: row i <-> col argsort(perm)[i]
        truth = np.argsort(perm)
        res.append(float((plan.argmax(1) == truth).mean()))  # still ground-truth-consistent
    return float(np.mean(res)), float(np.std(res))
 
 
# ---------- pipeline ----------
def run(Xa, Xb, dim, whiten, restarts, eps, k, shuffle_baseline=True):
    Za = preprocess(Xa, dim=dim, whiten=whiten)
    Zb = preprocess(Xb, dim=dim, whiten=whiten)
    C1, C2 = dist_matrix(Za), dist_matrix(Zb)
    plan, val, vals = gw_blind_match(C1, C2, n_restarts=restarts, entropic_eps=eps)
    res = retrieval_metrics(plan)
    res.update(n=int(Xa.shape[0]), gw_dist_best=val,
               gw_dist_restart_std=float(np.std(vals)),
               mutual_knn=mutual_knn(Za, Zb, k=min(k, Xa.shape[0] - 1)))
    if shuffle_baseline:
        m, s = permutation_null(C1, C2, n_perm=10)
        res["null_top1_mean"], res["null_top1_std"] = m, s
    return res
 
 
def synthetic(n=200, d_latent=32, noise=0.3, seed=0):
    """Two 'models' that see the same latent structure through different maps."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d_latent))
    Qa, _ = np.linalg.qr(rng.normal(size=(d_latent, d_latent)))
    Wb = rng.normal(size=(d_latent, 128))
    Xa = Z @ Qa + noise * rng.normal(size=(n, d_latent))
    Xb = np.tanh(Z @ Wb / np.sqrt(d_latent)) + noise * rng.normal(size=(n, 128))
    return Xa, Xb
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a"); ap.add_argument("--b")
    ap.add_argument("--level", choices=["class", "sample"], default="class")
    ap.add_argument("--n", type=int, default=1000, help="samples for sample-level")
    ap.add_argument("--dim", type=int, default=64, help="common PCA dim")
    ap.add_argument("--whiten", action="store_true")
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--eps", type=float, default=None, help="entropic GW epsilon")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
 
    if a.synthetic:
        Xa, Xb = synthetic(seed=a.seed)
    else:
        XA, LA = load(a.a); XB, LB = load(a.b)
        assert (LA == LB).all(), "samples must be in the same order in both files"
        if a.level == "class":
            Xa, keep = class_centroids(XA, LA)
            Xb, _ = class_centroids(XB, LB)
        else:
            idx = np.random.default_rng(a.seed).choice(len(XA), min(a.n, len(XA)), replace=False)
            Xa, Xb = XA[idx], XB[idx]
 
    res = run(Xa, Xb, a.dim, a.whiten, a.restarts, a.eps, a.k)
    print(json.dumps(res, indent=2))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
 