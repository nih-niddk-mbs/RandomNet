import sys; sys.path.insert(0, 'scripts')
import numpy as np
from random_network import make_weights

rng = np.random.default_rng(42)
sigma_c = 2 * np.pi
I, alpha, beta, dt = 1.0, 1.0, 1.0, 0.02
N = 256

def F(u_): return alpha * np.maximum(I + u_, 0.0) ** (1.0 / alpha)

for g in [1.5, 2.0]:
    sigma = g * sigma_c
    W = make_weights(N, sigma, rng=rng)
    phi = rng.uniform(-np.pi, np.pi, N)
    u = np.zeros(N)
    nan_triggered = 0
    print(f"\ng={g:.1f}:")
    print(f"  {'t':>4}  {'mean(u)':>10}  {'std(u)':>10}  {'max|u|':>10}  {'nan_count':>10}")
    for t_sec in range(60):
        for _ in range(int(1.0 / dt)):
            phi += F(u) * dt
            spikes = phi >= np.pi
            # Fixed: wrap phi correctly for multi-spike steps (F(u)*dt > 2*pi)
            phi[spikes] = ((phi[spikes] + np.pi) % (2.0 * np.pi)) - np.pi
            drive = W @ spikes.astype(float)
            u += -beta * u * dt + beta * drive
            if not np.all(np.isfinite(u)):
                nan_triggered += 1
                u = np.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)
        if t_sec < 5 or t_sec % 10 == 9:
            print(f"  {t_sec+1:4d}  {np.mean(u):10.2f}  {np.std(u):10.2f}  "
                  f"{np.max(np.abs(u)):10.2f}  {nan_triggered:10d}")
    sys.stdout.flush()
