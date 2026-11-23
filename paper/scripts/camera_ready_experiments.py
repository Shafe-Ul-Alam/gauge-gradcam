"""
camera_ready_experiments.py  --  ICCIT 2026 camera-ready (Paper #1827) reviewer experiments
==========================================================================================
All new experiments use the DEDUPLICATED brain-MRI file list (dedup_brain_mri.json,
built by dedup.py: exact-md5 + near-duplicate (32x32 Pearson r > 0.99) groups collapsed to
one image; label-conflicting groups dropped).

Experiments (pick with --exp):
  table1     R11.1  Table I main rows on the deduplicated split (pretrained ResNet-18)
  mlphead    R11.4  hidden FC layer before the classifier (Linear-ReLU-Linear head)
  serveradapt R11.5 FedAdam / FedYogi with server optimiser on parameters only, lr sweep
  ig         R2.3   Integrated Gradients on collapsed vs corrected vs healthy checkpoints

Usage (laptop GPU or CPU):
  python src_imaging/camera_ready_experiments.py --exp table1 --seeds 0 1 2
  python src_imaging/camera_ready_experiments.py --exp mlphead --seeds 0 1 2
Results are appended per run to results_imaging/camera_ready/<exp>.jsonl, so an
interrupted run resumes where it stopped (finished configs are skipped).
"""
import os, sys, json, copy, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image
import torchvision.transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "brain_mri")
OUT = os.path.join(ROOT, "results_imaging", "camera_ready")
CKPT = os.path.join(ROOT, "results_imaging", "model_cache_camera_ready")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMG, BS, N_CLIENTS, LOCAL_EPOCHS = 128, 16, 4, 1
N_PROBE = 30

# ------------------------------------------------------------------ data
_tf = T.Compose([T.Resize((IMG, IMG)), T.Grayscale(3), T.ToTensor(),
                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def load_dedup(max_per_class=None):
    meta = json.load(open(os.path.join(HERE, "dedup_brain_mri.json")))
    classes = meta["classes"]
    by = {c: [] for c in classes}
    for p in meta["kept_paths"]:
        by[p.split("/")[0]].append(p)
    files, labels = [], []
    for ci, c in enumerate(classes):
        ps = sorted(by[c])
        if max_per_class:
            ps = ps[:max_per_class]
        files += ps; labels += [ci] * len(ps)
    X = torch.zeros(len(files), 3, IMG, IMG)
    for i, p in enumerate(files):
        X[i] = _tf(Image.open(os.path.join(DATA, p)).convert("L"))
    return X, torch.tensor(labels), classes

def split(X, y, seed, test_frac=0.2):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y)); nt = int(len(y) * test_frac)
    return X[idx[nt:]], X[idx[:nt]], y[idx[nt:]], y[idx[:nt]]

def partition(y, mode, seed, alpha=0.1):
    y = np.asarray(y); rng = np.random.default_rng(seed)
    if mode == "centralized":
        return [np.arange(len(y))]
    if mode == "iid":
        return list(np.array_split(rng.permutation(len(y)), N_CLIENTS))
    parts = [[] for _ in range(N_CLIENTS)]
    for c in np.unique(y):
        ic = np.where(y == c)[0]; rng.shuffle(ic)
        cuts = (np.cumsum(rng.dirichlet([alpha] * N_CLIENTS)) * len(ic)).astype(int)[:-1]
        for i, ch in enumerate(np.split(ic, cuts)):
            parts[i] += ch.tolist()
    return [np.array(p, dtype=int) for p in parts]

# ------------------------------------------------------------------ model
def build(K, seed, pretrained=False, head="linear", hidden=256):
    torch.manual_seed(seed)
    m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    d = m.fc.in_features
    m.fc = nn.Linear(d, K) if head == "linear" else nn.Sequential(
        nn.Linear(d, hidden), nn.ReLU(inplace=False), nn.Linear(hidden, K))
    return m.to(DEV)

def final_linear(m):
    return m.fc if isinstance(m.fc, nn.Linear) else m.fc[-1]

def rho_of(m):
    W = final_linear(m).weight.detach().cpu(); K = W.shape[0]
    wb = W.mean(0); R = W - wb
    return float(wb.norm() / (R.norm() / np.sqrt(K) + 1e-12)), float((W < 0).float().mean())

def gauge_fix_(m):
    h = final_linear(m)
    with torch.no_grad():
        h.weight -= h.weight.mean(0, keepdim=True)
        if h.bias is not None:
            h.bias -= h.bias.mean()

# ------------------------------------------------------------------ train / eval
def local_train(m, X, y, idx, kind, lr):
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=lr) if kind == "adam" else torch.optim.SGD(m.parameters(), lr=lr)
    for _ in range(LOCAL_EPOCHS):
        perm = np.random.permutation(len(idx))
        for i in range(0, len(idx), BS):
            b = idx[perm[i:i + BS]]
            if len(b) < 2:            # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            F.cross_entropy(m(X[b].to(DEV)), y[b].to(DEV)).backward()
            opt.step()

def average(states, sizes):
    tot = float(sum(sizes)); out = {}
    for k in states[0]:
        if states[0][k].dtype.is_floating_point:
            out[k] = sum(s[k].float() * (n / tot) for s, n in zip(states, sizes)).to(states[0][k].dtype)
        else:
            out[k] = states[0][k].clone()
    return out

@torch.no_grad()
def evaluate(m, X, y, K):
    m.eval(); P = []
    for i in range(0, len(y), 64):
        P.append(torch.softmax(m(X[i:i + 64].to(DEV)), 1).cpu())
    P = torch.cat(P); pred = P.argmax(1); yn = y.numpy()
    acc = float((pred == y).float().mean())
    bal = float(np.mean([(pred[y == c] == c).float().mean() for c in range(K) if (y == c).any()]))
    maj = float(np.bincount(yn, minlength=K).max() / len(yn))
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(yn, P.numpy(), multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")
    return {"accuracy": acc, "balanced_accuracy": bal, "auc": auc, "majority": maj}

def gradcam_diag(m, X, n=N_PROBE):
    """Exact Grad-CAM at layer4 via autograd (valid for linear AND MLP heads)."""
    m.eval(); nd, pos, wneg = 0, [], []
    for i in range(min(n, len(X))):
        acts = {}
        h = m.layer4.register_forward_hook(lambda mod, inp, out: acts.__setitem__("a", out))
        x = X[i:i + 1].to(DEV).requires_grad_(False)
        logits = m(x); h.remove()
        t = int(logits.argmax(1))
        a = acts["a"]; g = torch.autograd.grad(logits[0, t], a)[0][0]
        w = g.mean(dim=(1, 2)); raw = (w[:, None, None] * a[0].detach()).sum(0)
        if float(F.relu(raw).max()) <= 1e-8:
            nd += 1
        pos.append(float((raw > 0).float().mean())); wneg.append(float((w < 0).float().mean()))
    return nd / n, float(np.mean(pos)), float(np.mean(wneg))

def federated(Xtr, ytr, K, seed, mode="noniid", kind="adam", lr=1e-3, gauge=False,
              rounds=40, pretrained=False, head="linear", server_opt="avg", server_lr=1e-2,
              tau=1e-3, alpha=0.1, log_every=0):
    np.random.seed(seed); torch.manual_seed(seed)
    parts = [p for p in partition(ytr, mode, seed, alpha) if len(p) > 1]
    gm = build(K, seed, pretrained, head)
    pnames = {n for n, _ in gm.named_parameters()}
    ms = {n: torch.zeros_like(p) for n, p in gm.named_parameters()}
    vs = {n: torch.zeros_like(p) for n, p in gm.named_parameters()}
    traj = []
    for r in range(1, rounds + 1):
        g = copy.deepcopy(gm.state_dict()); states, sizes = [], []
        for idx in parts:
            lm = build(K, seed, False, head); lm.load_state_dict(g)
            local_train(lm, Xtr, ytr, idx, kind, lr)
            states.append({k: v.detach().clone() for k, v in lm.state_dict().items()}); sizes.append(len(idx))
            del lm
        new = average(states, sizes)
        if server_opt != "avg":           # adaptive server step on PARAMETERS only; buffers averaged
            for k in pnames:
                d = (new[k] - g[k]).float()
                ms[k] = 0.9 * ms[k] + 0.1 * d
                if server_opt == "fedadam":
                    vs[k] = 0.99 * vs[k] + 0.01 * d * d
                else:  # fedyogi
                    vs[k] = vs[k] - 0.01 * d * d * torch.sign(vs[k] - d * d)
                new[k] = (g[k].float() + server_lr * ms[k] / (vs[k].clamp(min=0).sqrt() + tau)).to(g[k].dtype)
        gm.load_state_dict(new)
        if gauge:
            gauge_fix_(gm)
        if log_every and r % log_every == 0:
            traj.append({"round": r, "rho": rho_of(gm)[0]})
    return gm, traj

# ------------------------------------------------------------------ bookkeeping
def done_keys(path):
    if not os.path.exists(path):
        return set()
    return {json.loads(l)["key"] for l in open(path) if l.strip()}

def record(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")

def run_grid(exp, configs, seeds, max_per_class, pretrained, rounds):
    path = os.path.join(OUT, exp + ".jsonl"); done = done_keys(path)
    X, y, classes = load_dedup(max_per_class); K = len(classes)
    print(f"[data] dedup brain MRI: {len(y)} images, per class {np.bincount(y.numpy()).tolist()}", flush=True)
    for seed in seeds:
        Xtr, Xte, ytr, yte = split(X, y, seed)
        for name, kw in configs:
            key = f"{name}|s{seed}"
            if key in done:
                print("  [skip]", key); continue
            t0 = time.time()
            gm, traj = federated(Xtr, ytr, K, seed, rounds=rounds, pretrained=pretrained, **kw)
            ev = evaluate(gm, Xte, yte, K); deg, pos, wneg = gradcam_diag(gm, Xte)
            rho, fneg = rho_of(gm)
            row = {"key": key, "config": name, "seed": seed, **ev, "degeneracy": deg,
                   "positive_fraction": pos, "gradcam_w_negative": wneg, "rho": rho,
                   "frac_negative_W": fneg, "minutes": round((time.time() - t0) / 60, 1),
                   "n_train": int(len(ytr)), "n_test": int(len(yte)), "kw": kw}
            record(path, row)
            print(f"  {key:38s} acc {ev['accuracy']:.3f} bal {ev['balanced_accuracy']:.3f} "
                  f"deg {deg:.2f} rho {rho:.2f} ({row['minutes']} min)", flush=True)
            os.makedirs(CKPT, exist_ok=True)
            torch.save(gm.state_dict(), os.path.join(CKPT, f"{exp}_{name}_s{seed}.pt"))

EXPS = {
    # R11.1: Table I on deduplicated data, ImageNet-pretrained (paper's full-scale setting)
    "table1": dict(pretrained=True, max_per_class=None, rounds=40, configs=[
        ("noniid_adam",       dict(mode="noniid", kind="adam", lr=1e-3)),
        ("noniid_adam_gauge", dict(mode="noniid", kind="adam", lr=1e-3, gauge=True)),
        ("noniid_sgd",        dict(mode="noniid", kind="sgd",  lr=5e-2)),
        ("iid_adam",          dict(mode="iid",    kind="adam", lr=1e-3)),
        ("centralized_adam",  dict(mode="centralized", kind="adam", lr=1e-3)),
    ]),
    # R11.4: hidden FC layer before the classifier (characterisation setting: scratch, 400/class)
    "mlphead": dict(pretrained=False, max_per_class=400, rounds=40, configs=[
        ("mlp_noniid_adam",       dict(mode="noniid", kind="adam", lr=1e-3, head="mlp")),
        ("mlp_noniid_adam_gauge", dict(mode="noniid", kind="adam", lr=1e-3, head="mlp", gauge=True)),
        ("mlp_noniid_sgd",        dict(mode="noniid", kind="sgd",  lr=5e-2, head="mlp")),
    ]),
    # R11.1 fallback: characterisation setting (scratch, 400/class) on the deduplicated set
    "dedup_char": dict(pretrained=False, max_per_class=400, rounds=40, configs=[
        ("noniid_adam",       dict(mode="noniid", kind="adam", lr=1e-3)),
        ("noniid_adam_gauge", dict(mode="noniid", kind="adam", lr=1e-3, gauge=True)),
        ("noniid_sgd",        dict(mode="noniid", kind="sgd",  lr=5e-2)),
        ("iid_adam",          dict(mode="iid",    kind="adam", lr=1e-3)),
    ]),
    # R11.5: server-side adaptive optimisers, server lr sweep (short), then full runs
    "serveradapt_sweep": dict(pretrained=False, max_per_class=400, rounds=15, configs=[
        (f"{so}_cSGD_slr{slr}", dict(mode="noniid", kind="sgd", lr=5e-2, server_opt=so, server_lr=slr))
        for so in ["fedadam"] for slr in [1e-3, 3e-3, 1e-2, 3e-2]]),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--server_lr", type=float, default=None, help="for --exp serveradapt")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    print(f"[device] {DEV}", flush=True)
    if a.exp == "serveradapt":
        slr = a.server_lr
        cfg = dict(pretrained=False, max_per_class=400, rounds=40, configs=[
            ("fedavg_cSGD",            dict(mode="noniid", kind="sgd",  lr=5e-2)),
            (f"fedadam_cSGD",          dict(mode="noniid", kind="sgd",  lr=5e-2, server_opt="fedadam", server_lr=slr)),
            (f"fedyogi_cSGD",          dict(mode="noniid", kind="sgd",  lr=5e-2, server_opt="fedyogi", server_lr=slr)),
            (f"fedadam_cAdam",         dict(mode="noniid", kind="adam", lr=1e-3, server_opt="fedadam", server_lr=slr)),
            (f"fedadam_cAdam_gauge",   dict(mode="noniid", kind="adam", lr=1e-3, server_opt="fedadam", server_lr=slr, gauge=True)),
        ])
    else:
        cfg = EXPS[a.exp]
    run_grid(a.exp, cfg["configs"], a.seeds, cfg["max_per_class"], cfg["pretrained"], cfg["rounds"])

if __name__ == "__main__":
    main()
