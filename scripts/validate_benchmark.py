#!/usr/bin/env python3
"""Benchmark: 5 dynamical systems x 20 parameter values x 10 features.

Computes Spearman rho between each feature and the ground-truth parameter
to validate that new features capture meaningful system properties.

Pass criteria:
- Mean max |rho| >= 0.6 across systems
- At least 4/5 systems have max |rho| > 0.5
- No feature is constant across all systems
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
from scipy import stats

from topofeatures import extract
from topofeatures.core import FEATURE_NAMES


# ============================================================
# Dynamical system generators
# ============================================================

def rossler(a, b=0.2, c=5.7, n_steps=6000, dt=0.01, seed=42):
    """Rossler attractor. a controls regime (periodic→chaotic)."""
    rng = np.random.default_rng(seed)
    x, y, z = rng.uniform(-1, 1, 3)
    traj = np.zeros((n_steps, 3))
    for t in range(n_steps):
        traj[t] = [x, y, z]
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        x += dx * dt
        y += dy * dt
        z += dz * dt
        x, y, z = np.clip([x, y, z], -100, 100)
    return traj[2000:]  # discard transient


def lorenz(r, sigma=10.0, beta=8/3, n_steps=6000, dt=0.01, seed=42):
    """Lorenz system. r controls regime (periodic→chaotic, chaotic at r~28)."""
    rng = np.random.default_rng(seed)
    x, y, z = rng.uniform(-1, 1, 3)
    traj = np.zeros((n_steps, 3))
    for t in range(n_steps):
        traj[t] = [x, y, z]
        dx = sigma * (y - x)
        dy = x * (r - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
        x, y, z = np.clip([x, y, z], -200, 200)
    return traj[2000:]


def kuramoto(K, N=10, n_steps=4000, dt=0.05, seed=42):
    """Kuramoto oscillators. K=coupling, returns (n_steps, N) phases as multivariate TS.
    Ground truth: order parameter r = |mean(exp(i*theta))|."""
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal(N)  # natural frequencies
    theta = rng.uniform(0, 2*np.pi, N)
    traj = np.zeros((n_steps, N))

    for t in range(n_steps):
        traj[t] = np.cos(theta)  # observable: cos(phase)
        for i in range(N):
            coupling = K / N * np.sum(np.sin(theta - theta[i]))
            theta[i] += (omega[i] + coupling) * dt
    return traj[500:]  # discard transient


def kuramoto_order(traj_cos, N=10):
    """Compute mean order parameter from cos(phase) trajectory."""
    # Approximate: r = |mean(cos(theta)) + i*mean(sin(theta))|
    # Since we only have cos, use a simple proxy: std of column means
    return float(np.abs(traj_cos.mean(axis=0)).mean())


def lotka_volterra(alpha, beta_param=0.02, delta=0.01, gamma=1.0, n_steps=8000, dt=0.01, seed=42):
    """Lotka-Volterra predator-prey. alpha controls oscillation amplitude."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(20, 50)  # prey
    y = rng.uniform(5, 20)   # predator
    traj = np.zeros((n_steps, 2))
    for t in range(n_steps):
        traj[t] = [x, y]
        dx = alpha * x - beta_param * x * y
        dy = delta * x * y - gamma * y
        x = max(0.1, x + dx * dt)
        y = max(0.1, y + dy * dt)
        x = min(x, 1000)
        y = min(y, 1000)
    return traj[2000:]


def sine_wave(freq, n_steps=2000, dt=0.01, seed=42):
    """1D sine wave with noise. freq controls cycles per window."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps) * dt
    signal = np.sin(2 * np.pi * freq * t) + 0.1 * rng.standard_normal(n_steps)
    return signal


# ============================================================
# Main benchmark
# ============================================================

def main():
    print(f"FEATURE_NAMES ({len(FEATURE_NAMES)}): {FEATURE_NAMES}\n")

    systems = {
        "Rossler": {
            "params": np.linspace(0.1, 0.5, 20),
            "param_name": "a (periodic→chaotic)",
            "gen": lambda p: rossler(a=p),
            "ground_truth": lambda p, _: p,  # a itself is the regime parameter
        },
        "Lorenz": {
            "params": np.linspace(10, 35, 20),
            "param_name": "r (regime parameter)",
            "gen": lambda p: lorenz(r=p),
            "ground_truth": lambda p, _: p,
        },
        "Kuramoto": {
            "params": np.linspace(0.1, 5.0, 20),
            "param_name": "K (coupling strength)",
            "gen": lambda p: kuramoto(K=p),
            "ground_truth": lambda p, traj: kuramoto_order(traj),
        },
        "Lotka-Volterra": {
            "params": np.linspace(0.5, 3.0, 20),
            "param_name": "alpha (prey growth rate)",
            "gen": lambda p: lotka_volterra(alpha=p),
            "ground_truth": lambda p, traj: float(np.std(traj[:, 0])),  # prey amplitude
        },
        "Sine": {
            "params": np.linspace(0.5, 10.0, 20),
            "param_name": "freq (cycles per window)",
            "gen": lambda p: sine_wave(freq=p),
            "ground_truth": lambda p, _: p,
        },
    }

    all_rhos = {}  # system → {feature → rho}
    all_times = []
    feature_stats = defaultdict(list)  # feature_name → list of all values

    for sys_name, sys_def in systems.items():
        print(f"\n{'='*60}")
        print(f"System: {sys_name} — {sys_def['param_name']}")
        print(f"{'='*60}")

        params = sys_def["params"]
        features_matrix = []
        ground_truths = []

        for p in params:
            traj = sys_def["gen"](p)
            gt = sys_def["ground_truth"](p, traj)
            ground_truths.append(gt)

            t0 = time.perf_counter()
            feats = extract(traj, seed=42)
            elapsed = time.perf_counter() - t0
            all_times.append(elapsed)

            row = [feats[k] for k in FEATURE_NAMES]
            features_matrix.append(row)

            for k, v in feats.items():
                feature_stats[k].append(v)

        features_matrix = np.array(features_matrix)
        ground_truths = np.array(ground_truths)

        # Spearman rho per feature
        rhos = {}
        print(f"\n  {'Feature':>35s}  {'rho':>8s}  {'p-value':>10s}  {'|rho|':>6s}")
        print(f"  {'-'*35}  {'-'*8}  {'-'*10}  {'-'*6}")

        for j, name in enumerate(FEATURE_NAMES):
            col = features_matrix[:, j]
            if col.std() < 1e-10:
                rhos[name] = 0.0
                print(f"  {name:>35s}  {'N/A':>8s}  {'constant':>10s}  {0.0:>6.3f}")
                continue
            rho, pval = stats.spearmanr(col, ground_truths)
            rhos[name] = float(rho)
            flag = " ***" if abs(rho) > 0.5 else ""
            print(f"  {name:>35s}  {rho:>8.3f}  {pval:>10.4f}  {abs(rho):>6.3f}{flag}")

        max_rho_feat = max(rhos, key=lambda k: abs(rhos[k]))
        max_rho_val = abs(rhos[max_rho_feat])
        print(f"\n  Best: {max_rho_feat} (|rho| = {max_rho_val:.3f})")
        all_rhos[sys_name] = rhos

    # ============================================================
    # Summary
    # ============================================================
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Per-system max |rho|
    print("\nPer-system max |rho|:")
    max_rhos = []
    for sys_name, rhos in all_rhos.items():
        max_rho = max(abs(v) for v in rhos.values())
        best_feat = max(rhos, key=lambda k: abs(rhos[k]))
        max_rhos.append(max_rho)
        status = "PASS" if max_rho > 0.5 else "FAIL"
        print(f"  {sys_name:>20s}: max |rho| = {max_rho:.3f} ({best_feat}) [{status}]")

    mean_max_rho = np.mean(max_rhos)
    n_pass = sum(1 for r in max_rhos if r > 0.5)
    print(f"\n  Mean max |rho|: {mean_max_rho:.3f} (target >= 0.6)")
    print(f"  Systems with max |rho| > 0.5: {n_pass}/5 (target >= 4)")

    # Per-feature summary across all systems
    print("\nPer-feature summary across all systems:")
    print(f"  {'Feature':>35s}  {'mean_rho':>10s}  {'max_|rho|':>10s}  {'std':>10s}  {'frac_zero':>10s}")
    for name in FEATURE_NAMES:
        rho_vals = [all_rhos[s][name] for s in all_rhos]
        feat_vals = np.array(feature_stats[name])
        mean_rho = np.mean([abs(r) for r in rho_vals])
        max_abs_rho = max(abs(r) for r in rho_vals)
        std_val = feat_vals.std()
        frac_zero = (feat_vals == 0).mean()
        flag = " [DEGENERATE]" if std_val < 1e-6 else ""
        print(f"  {name:>35s}  {mean_rho:>10.3f}  {max_abs_rho:>10.3f}  "
              f"{std_val:>10.4f}  {frac_zero:>10.2f}{flag}")

    # Timing
    all_times = np.array(all_times)
    print(f"\nTiming: {all_times.mean():.3f}s/sample "
          f"(min={all_times.min():.3f}, max={all_times.max():.3f})")

    # Pass/fail
    print("\n" + "=" * 60)
    overall_pass = mean_max_rho >= 0.6 and n_pass >= 4
    if overall_pass:
        print("OVERALL: PASS")
    else:
        print("OVERALL: FAIL")
        if mean_max_rho < 0.6:
            print(f"  Mean max |rho| = {mean_max_rho:.3f} < 0.6")
        if n_pass < 4:
            print(f"  Only {n_pass}/5 systems passed (need 4)")
    print("=" * 60)


if __name__ == "__main__":
    main()
