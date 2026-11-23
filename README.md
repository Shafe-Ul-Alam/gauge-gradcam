# gauge-gradcam

Weight-only diagnostic and one-line server-side fix for **silent Grad-CAM collapse in federated learning**.

Companion code for *On the Collapse of Grad-CAM under Federated Optimization: A Gauge-Theoretic
Diagnosis and Correction* (IEEE ICCIT 2026).

Under client label skew, adaptive client optimisers (Adam, AdamW, RMSProp, Adagrad) make the mean row
`w̄` of the classifier head drift. `w̄` is a gauge freedom (it adds the same constant to every logit, so
predictions never change), but Grad-CAM reads the head weights directly, so a large negative `w̄` makes
every channel weight negative and the ReLU returns an all-zero map.

## Audit any checkpoint (no data, no forward pass)

```bash
python gauge_gradcam.py model.pt            # prints rho; rho > 1  =>  Grad-CAM maps unreliable
```

`rho = ||w̄|| / (||W - 1 w̄ᵀ||_F / sqrt(K))` is scale-invariant and separated collapsed from healthy models
with AUC 1.000 over 55 checkpoints in the paper.

## Fix: gauge-normalised FedAvg

```python
from gauge_gradcam import fedavg_round, gauge_normalise_
rho_before = fedavg_round(global_model, client_state_dicts, client_sizes, gauge=True)
# or, on any trained model:  gauge_normalise_(model)
```

The correction subtracts the mean row and mean bias from the head after aggregation. Logits shift by a
constant, so predictions and probabilities are unchanged; no client change and no extra communication.

## Demo

```bash
python demo.py
```

## Reproducing the paper

The full experiment scripts (`src_imaging/`) and per-seed result files are in `paper/`.
The brain MRI data is the Kaggle *Brain Tumor MRI Dataset* (M. Nickparvar); `paper/dedup_brain_mri.json`
lists the 5,809 images kept after near-duplicate removal.

## License

MIT

### Camera-ready experiments (reviewer responses)

| Script | What it does |
|---|---|
| `paper/scripts/dedup.py` | near-duplicate audit of the brain MRI release; writes `dedup_brain_mri.json` |
| `paper/scripts/camera_ready_experiments.py --exp table1` | Table I rows on the deduplicated set (ImageNet-pretrained ResNet-18) |
| `... --exp mlphead` | hidden FC layer before the classifier |
| `... --exp serveradapt --server_lr 0.03` | FedAdam / FedYogi (server step on parameters only) |
| `paper/scripts/drift_check.py` | how closely the drift law matches measured drift |
| `paper/scripts/ig_check.py` | Integrated Gradients on collapsed / corrected / healthy checkpoints |

Per-run results are in `paper/results/*.jsonl`.
