"""Minimal demo: federated linear head on synthetic non-negative features with label skew.
Adam clients make rho grow; gauge normalisation keeps it at zero with identical predictions."""
import torch, torch.nn as nn, torch.nn.functional as F
from gauge_gradcam import rho, fedavg_round

torch.manual_seed(0)
K, C, N = 4, 64, 400
centers = torch.rand(K, C) * 2
X = torch.relu(centers.repeat_interleave(N, 0) + 0.5 * torch.randn(K * N, C))   # non-negative, like post-ReLU features
y = torch.arange(K).repeat_interleave(N)
clients = [torch.where(y == k)[0] for k in range(K)]                              # one class per client (extreme skew)

for gauge in [False, True]:
    torch.manual_seed(0)
    g = nn.Linear(C, K)
    for rnd in range(30):
        states, sizes = [], []
        for idx in clients:
            m = nn.Linear(C, K); m.load_state_dict(g.state_dict())
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            for _ in range(5):
                opt.zero_grad(); F.cross_entropy(m(X[idx]), y[idx]).backward(); opt.step()
            states.append(m.state_dict()); sizes.append(len(idx))
        r = fedavg_round(g, states, sizes, gauge=gauge)
    acc = (g(X).argmax(1) == y).float().mean().item()
    print(f"gauge={gauge}:  rho before last normalisation = {r:.2f},  rho now = {rho(g.weight):.2f},  "
          f"frac negative weights = {(g.weight < 0).float().mean():.2f},  accuracy = {acc:.3f}")
