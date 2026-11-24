"""
drift_check.py

How closely does the drift law (Eq. 5, a full-step approximation) predict the measured
per-round growth of ||w_bar|| for Adam clients? Federated linear probes on frozen features
of a randomly initialised ResNet-18 (the probe setting of Fig. 3(b)), one class per client.

Features are either loaded from a cache (--features, a .pt file with ftr/ytr) or extracted
from the brain MRI folder (--data_dir): first 400 images per class (sorted file names),
80/20 split with numpy default_rng(0).

Usage:
  python drift_check.py --data_dir /path/to/brain_mri --out drift_check.json
"""
import os, json, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image


def extract_features(data_dir, per_class=400, img=128, seed=0):
    tf = T.Compose([T.Resize((img, img)), T.Grayscale(3), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    classes = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    X, y = [], []
    for ci, c in enumerate(classes):
        files = sorted(f for f in os.listdir(os.path.join(data_dir, c))
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))[:per_class]
        for f in files:
            X.append(tf(Image.open(os.path.join(data_dir, c, f)).convert("L")))
            y.append(ci)
    X, y = torch.stack(X), torch.tensor(y)
    idx = np.random.default_rng(seed).permutation(len(y))
    tr = idx[int(0.2 * len(y)):]
    torch.manual_seed(seed)
    net = tvm.resnet18(weights=None)
    net.fc = nn.Identity()
    net.eval()
    with torch.no_grad():
        f = torch.cat([net(X[tr][i:i + 64]) for i in range(0, len(tr), 64)])
    return F.relu(f), y[tr]


def run(ftr, ytr, K, seed, eta=1e-3, S=5, M=4, rounds=30):
    """One federated linear-probe run; returns slope of ||w_bar|| and realised step sizes."""
    torch.manual_seed(seed)
    y = np.asarray(ytr)
    if K < 4:
        y = np.where(y <= K - 2, y, K - 1)
    rng = np.random.default_rng(seed)
    by = {c: list(rng.permutation(np.where(y == c)[0])) for c in range(K)}
    holders = {c: [i for i in range(M) if i % K == c] for c in range(K)}
    parts = [[] for _ in range(M)]
    for c, ix in by.items():
        for h, chunk in zip(holders[c], np.array_split(np.array(ix), len(holders[c]))):
            parts[h] += chunk.tolist()
    parts = [torch.tensor(sorted(p)) for p in parts if p]
    yt = torch.tensor(y)
    D = ftr.shape[1]
    g = torch.zeros(K, D)
    nn.init.normal_(g, std=0.01)
    norms, own, other, rowsum = [], [], [], []
    for _ in range(rounds):
        states = []
        for idx in parts:
            w = g.clone().requires_grad_(True)
            opt = torch.optim.Adam([w], lr=eta)
            k = int(np.bincount(y[idx.numpy()]).argmax())
            for _ in range(S):
                w0 = w.detach().clone()
                opt.zero_grad()
                F.cross_entropy(ftr[idx] @ w.T, yt[idx]).backward()
                opt.step()
                d = w.detach() - w0
                own.append(float(d[k].mean() / eta))
                other.append(float(torch.cat([d[j:j + 1] for j in range(K) if j != k]).mean() / eta))
                rowsum.append(float(d.sum(0).mean() / eta))
            states.append(w.detach().clone())
        g = torch.stack(states).mean(0)
        norms.append(float(g.mean(0).norm()))
    h = rounds // 2
    slope = (norms[-1] - norms[h]) / (rounds - h)
    return slope, np.mean(own), np.mean(other), np.mean(rowsum), D, S, eta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--features", default=None, help="cached .pt with keys ftr, ytr")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="drift_check.json")
    a = ap.parse_args()
    if a.features:
        d = torch.load(a.features, map_location="cpu")
        ftr, ytr = d["ftr"], d["ytr"]
    else:
        ftr, ytr = extract_features(a.data_dir)
    out = {}
    for K in [2, 3, 4]:
        rows = [run(ftr, ytr, K, s) for s in a.seeds]
        slope, own, other, rowsum = (float(np.mean([r[i] for r in rows])) for i in range(4))
        D, S, eta = rows[0][4], rows[0][5], rows[0][6]
        pred_literal = S * (K - 2) * eta * np.sqrt(D) / K
        pred_realised = abs(rowsum) * eta * S * np.sqrt(D) / K
        out[f"K={K}"] = {"measured_slope": slope, "pred_eq5_literal": pred_literal,
                         "realised_step_own_row": own, "realised_step_other_rows": other,
                         "realised_row_sum_per_step": rowsum, "pred_realised_steps": pred_realised}
        print(f"K={K}: measured {slope:.4f}, Eq.5 literal {pred_literal:.4f}, "
              f"with realised steps {pred_realised:.4f}")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
