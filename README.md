# gauge-gradcam

Weight-only diagnostic and server-side correction for **silent Grad-CAM collapse in federated
learning**.

Companion code for *On the Collapse of Grad-CAM under Federated Optimization: A Gauge-Theoretic
Diagnosis and Correction* (IEEE ICCIT 2026).

## Background

Softmax cross-entropy is invariant to adding the same vector to every row of the classifier head,
so the mean row `w̄` cannot change any prediction. Its gradient is zero, so SGD, momentum and
federated averaging conserve it, but adaptive client optimisers (Adam, AdamW, RMSProp, Adagrad)
under label skew drive it steadily negative. Grad-CAM reads the head weights directly, so a large
negative `w̄` makes every channel weight negative and the ReLU returns an all-zero map, while
accuracy and calibration are unaffected.

## Contents

| Path | Description |
|---|---|
| `gauge_gradcam.py` | `rho(W)` offset ratio, `gauge_normalise_(model)`, `fedavg_round(...)` with optional gauge normalisation, and a command-line checkpoint audit |
| `demo.py` | self-contained demo on synthetic features: Adam clients make `rho` grow; gauge normalisation keeps it at zero with identical predictions |
| `paper/dedup_brain_mri.json` | the 5,809 brain MRI images kept after near-duplicate removal |
| `paper/scripts/` | scripts for the experiments added in the camera-ready revision (see below) |
| `paper/results/` | per-seed results of those experiments (`*.jsonl`, `*.json`) |

The experiments of the original submission (Figs. 2–4, Tables II–III) are not included in this
repository.

## Installation

```bash
pip install -r requirements.txt     # torch; the paper scripts also need torchvision, numpy, scipy, scikit-learn, pillow
```

## Audit a checkpoint (no data, no forward pass)

```bash
python gauge_gradcam.py model.pt              # prints rho; rho > 1 flags unreliable saliency maps
python gauge_gradcam.py model.pt --key fc.weight
```

`rho = ||w̄|| / (||W - 1 w̄ᵀ||_F / sqrt(K))` is scale-invariant. The threshold `rho > 1` was fixed
before evaluation; in the paper it separated 17 collapsed from 38 healthy checkpoints without error.

## Correction: gauge-normalised FedAvg

```python
from gauge_gradcam import fedavg_round, gauge_normalise_

rho_before = fedavg_round(global_model, client_state_dicts, client_sizes, gauge=True)
# or, post hoc on any trained model:
gauge_normalise_(model)
```

The correction subtracts the mean row and mean bias from the classifier head. Every logit shifts
by the same constant, so predictions and probabilities are unchanged. Because no gradient depends
on `w̄`, training with the correction yields the same predictions as training without it
(Proposition 2 in the paper).

## Reproducing the camera-ready experiments

Data: Kaggle *Brain Tumor MRI Dataset* (M. Nickparvar), 7,200-image release, arranged as
`brain_mri/{glioma,meningioma,notumor,pituitary}/`.

```bash
cd paper/scripts
python dedup.py /path/to/brain_mri                                    # near-duplicate audit
python camera_ready_experiments.py --exp table1 --seeds 0 1 2 3 4 5 6 7 8 9 --data_dir /path/to/brain_mri
python camera_ready_experiments.py --exp mlphead --seeds 0 1 2 --data_dir /path/to/brain_mri
python camera_ready_experiments.py --exp serveradapt_sweep --seeds 0 --data_dir /path/to/brain_mri
python camera_ready_experiments.py --exp serveradapt --server_lr 0.03 --seeds 0 1 2 --data_dir /path/to/brain_mri
python drift_check.py --data_dir /path/to/brain_mri
python ig_check.py --data_dir /path/to/brain_mri \
    --ckpt collapsed=checkpoints/table1_noniid_adam_s0.pt gauge=checkpoints/table1_noniid_adam_gauge_s0.pt
```

`table1` uses ImageNet-pretrained ResNet-18 (torchvision downloads the weights on first use); one
run takes about 8 minutes on a GTX 1650. Runs are resumable: finished seeds are skipped. Results
are written to `paper/scripts/results/` unless `--out_dir` is given. GPU kernels are not
deterministic, so accuracies can differ slightly between runs; degeneracy and `rho` are stable.

## Citation

```
M. S. U. Alam and A. A. Rakin, "On the Collapse of Grad-CAM under Federated Optimization:
A Gauge-Theoretic Diagnosis and Correction," in Proc. IEEE ICCIT, 2026.
```

## License

MIT
