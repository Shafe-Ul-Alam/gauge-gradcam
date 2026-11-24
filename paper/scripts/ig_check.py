"""
ig_check.py

Integrated Gradients on the target logit splits exactly into a class-independent offset term
IG[w_bar . a] and a class-specific residual term IG[R_t . a]. This script measures, for each
checkpoint, the offset term's share of attribution mass and the mean correlation between the
IG maps of the four target classes (class-agnostic maps correlate highly).

Checkpoints are ResNet-18 state_dicts with a 4-way linear head (fc). Test images are drawn
from the deduplicated brain MRI set (split seed 123).

Usage:
  python ig_check.py --data_dir /path/to/brain_mri \
      --ckpt collapsed=noniid_adam.pt gauge=noniid_adam_gauge.pt healthy=iid_adam.pt
"""
import os, sys, json, argparse
import numpy as np
import torch, torch.nn as nn
import torchvision.models as tvm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera_ready_experiments as E


def integrated_gradients(fn, x, steps=32):
    alphas = torch.linspace(1.0 / steps, 1.0, steps).view(-1, 1, 1, 1)
    xs = (alphas * x).requires_grad_(True)            # zero baseline
    grads = torch.autograd.grad(fn(xs).sum(), xs)[0]
    return (x * grads.mean(0, keepdim=True))[0].sum(0)  # H x W, summed over channels


def analyse(state_dict, X, steps=32):
    m = tvm.resnet18(weights=None)
    m.fc = nn.Linear(512, 4)
    m.load_state_dict(state_dict)
    m.eval()
    W = m.fc.weight.detach()
    wbar = W.mean(0)
    R = W - wbar
    feat = nn.Sequential(*list(m.children())[:-1], nn.Flatten())
    shares, class_corr = [], []
    for i in range(len(X)):
        x = X[i:i + 1]
        with torch.no_grad():
            t = int(m(x).argmax())
        a_off = integrated_gradients(lambda z: feat(z) @ wbar, x, steps)
        a_res = [integrated_gradients(lambda z, k=k: feat(z) @ R[k], x, steps) for k in range(4)]
        shares.append(float(a_off.abs().sum() / (a_off.abs().sum() + a_res[t].abs().sum())))
        maps = [(a_off + a_res[k]).flatten().numpy() for k in range(4)]
        class_corr.append(np.mean([np.corrcoef(maps[p], maps[q])[0, 1]
                                   for p in range(4) for q in range(p + 1, 4)]))
    rho = float(wbar.norm() / (R.norm() / 2.0))
    return {"rho": rho, "offset_share_of_IG_mass": float(np.mean(shares)),
            "mean_corr_between_class_maps": float(np.mean(class_corr))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--n_images", type=int, default=24)
    ap.add_argument("--out", default="ig_check.json")
    a = ap.parse_args()
    E.DATA = a.data_dir
    X, y, _ = E.load_dedup(None)
    _, Xte, _, _ = E.split(X, y, 123)
    Xte = Xte[:a.n_images]
    res = {}
    for item in a.ckpt:
        name, path = item.split("=", 1)
        res[name] = analyse(torch.load(path, map_location="cpu"), Xte)
        print(name, {k: round(v, 3) for k, v in res[name].items()})
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
