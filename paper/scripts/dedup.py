"""
dedup.py

Near-duplicate audit of the Kaggle Brain Tumor MRI dataset (7,200-image release) and the
deduplicated file list used in the camera-ready experiments.

Two images are grouped if they are byte-identical (md5) or if their 32x32 grayscale,
mean-removed versions have Pearson correlation above 0.99. Each group keeps one image (the
first by file name); groups containing more than one label are dropped entirely.

Usage:
  python dedup.py /path/to/brain_mri      # writes dedup_brain_mri.json in the working dir
"""
import os, sys, json, hashlib
import numpy as np
from PIL import Image

root = sys.argv[1] if len(sys.argv) > 1 else "brain_mri"
THR = 0.99
classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
paths, labels = [], []
for ci, c in enumerate(classes):
    for f in sorted(os.listdir(os.path.join(root, c))):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            paths.append(f"{c}/{f}"); labels.append(ci)
labels = np.array(labels); n = len(paths)
X = np.zeros((n, 1024), np.float32); md5 = []
for i, p in enumerate(paths):
    fp = os.path.join(root, p)
    md5.append(hashlib.md5(open(fp, "rb").read()).hexdigest())
    a = np.asarray(Image.open(fp).convert("L").resize((32, 32), Image.BILINEAR), np.float32).ravel()
    a -= a.mean(); X[i] = a / (np.linalg.norm(a) + 1e-8)
par = list(range(n))
def find(a):
    while par[a] != a:
        par[a] = par[par[a]]; a = par[a]
    return a
def union(a, b): par[find(a)] = find(b)
seen = {}
for i, h in enumerate(md5):
    if h in seen: union(i, seen[h])
    else: seen[h] = i
n_exact_red = n - len(set(md5))
S = X @ X.T; np.fill_diagonal(S, -1)
I, J = np.nonzero(np.triu(S > THR, 1))
for a, b in zip(I, J): union(a, b)
groups = {}
for i in range(n): groups.setdefault(find(i), []).append(i)
keep, conflict_imgs, n_groups_dup = [], 0, 0
for g in groups.values():
    if len(g) > 1: n_groups_dup += 1
    if len(set(labels[g])) > 1:
        conflict_imgs += len(g); continue          # drop label-conflicting groups entirely
    keep.append(sorted(g)[0])
keep = sorted(keep)
split = np.array([p.split("/")[1][:2] for p in paths])
rep = {
  "n_images": n, "threshold_corr32": THR,
  "exact_md5_redundant": int(n_exact_red),
  "near_dup_pairs": int(len(I)), "near_dup_pairs_cross_TrTe": int((split[I] != split[J]).sum()),
  "near_dup_pairs_cross_label": int((labels[I] != labels[J]).sum()),
  "dup_groups": n_groups_dup, "label_conflict_images_dropped": int(conflict_imgs),
  "n_kept": len(keep), "n_removed": n - len(keep),
  "kept_per_class": {c: int((labels[keep] == ci).sum()) for ci, c in enumerate(classes)},
}
print(json.dumps(rep, indent=1))
json.dump({"report": rep, "classes": classes, "kept_paths": [paths[i] for i in keep]},
          open("dedup_brain_mri.json", "w"), indent=0)
