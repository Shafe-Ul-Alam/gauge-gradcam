"""
gauge_gradcam: weight-only Grad-CAM collapse diagnostic and gauge-normalised FedAvg.

Companion code for "On the Collapse of Grad-CAM under Federated Optimization:
A Gauge-Theoretic Diagnosis and Correction" (IEEE ICCIT 2026).

    rho(W)            offset ratio  ||w_bar|| / (||W - 1 w_bar^T||_F / sqrt(K))
    gauge_normalise_  remove the mean classifier row (prediction-preserving)
    fedavg_round      one FedAvg aggregation with optional gauge normalisation
"""
import math
import torch
import torch.nn as nn


def rho(W: torch.Tensor) -> float:
    """Offset ratio of a K x C classifier weight matrix. rho > 1 flags Grad-CAM collapse."""
    W = W.detach().float()
    K = W.shape[0]
    wbar = W.mean(0)
    R = W - wbar
    return float(wbar.norm() / (R.norm() / math.sqrt(K) + 1e-12))


def find_head(model: nn.Module) -> nn.Linear:
    """Return the last nn.Linear module (the classifier head)."""
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise ValueError("model has no nn.Linear layer")
    return last


@torch.no_grad()
def gauge_normalise_(model_or_head: nn.Module) -> None:
    """Subtract the mean row (and mean bias) from the classifier head, in place.
    Every logit shifts by the same constant, so softmax outputs and predictions
    are unchanged; Grad-CAM channel weights stop sharing a common offset."""
    head = model_or_head if isinstance(model_or_head, nn.Linear) else find_head(model_or_head)
    head.weight -= head.weight.mean(0, keepdim=True)
    if head.bias is not None:
        head.bias -= head.bias.mean()


@torch.no_grad()
def fedavg_round(global_model: nn.Module, client_states, client_sizes, gauge: bool = True):
    """Size-weighted FedAvg of client state_dicts, then (optionally) gauge normalisation.
    Returns the offset ratio of the aggregated head *before* normalisation."""
    total = float(sum(client_sizes))
    avg = {}
    for k, v in client_states[0].items():
        if v.dtype.is_floating_point:
            avg[k] = sum(s[k].float() * (n / total) for s, n in zip(client_states, client_sizes)).to(v.dtype)
        else:
            avg[k] = v.clone()
    global_model.load_state_dict(avg)
    r = rho(find_head(global_model).weight)
    if gauge:
        gauge_normalise_(global_model)
    return r


def audit_checkpoint(path: str, key: str = None) -> dict:
    """Load a state_dict (.pt) and report rho of its classifier head."""
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if key is None:
        cands = [k for k, v in sd.items() if k.endswith("weight") and getattr(v, "dim", lambda: 0)() == 2]
        key = cands[-1]
    r = rho(sd[key])
    return {"checkpoint": path, "head": key, "rho": round(r, 4), "flag": r > 1.0}


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Weight-only Grad-CAM collapse audit (rho > 1 => maps unreliable)")
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--key", default=None, help="state_dict key of the head weight (default: last 2-D weight)")
    a = ap.parse_args()
    for p in a.checkpoints:
        print(json.dumps(audit_checkpoint(p, a.key)))
