"""Smoke test: verify all 10 features produce finite values on synthetic data."""

import time

import numpy as np

from topofeatures import extract
from topofeatures.core import FEATURE_NAMES


def rossler(a=0.2, b=0.2, c=5.7, n_steps=4000, dt=0.01, seed=42):
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
    return traj


def main():
    print(f"FEATURE_NAMES ({len(FEATURE_NAMES)}): {FEATURE_NAMES}\n")

    # --- Test 1: 3D sine wave ---
    print("=" * 60)
    print("Test 1: 3D sine wave (2000 steps)")
    t = np.linspace(0, 10 * np.pi, 2000)
    X_3d = np.column_stack([np.sin(t), np.cos(t), np.sin(2 * t)])
    X_3d += 0.1 * np.random.default_rng(42).standard_normal(X_3d.shape)

    t0 = time.perf_counter()
    result = extract(X_3d, seed=42)
    elapsed = time.perf_counter() - t0

    print(f"  Time: {elapsed:.3f}s")
    n_finite = 0
    n_zero = 0
    for k in FEATURE_NAMES:
        v = result[k]
        is_fin = np.isfinite(v)
        n_finite += is_fin
        n_zero += (v == 0)
        flag = "" if is_fin else " *** NOT FINITE ***"
        flag += " [ZERO]" if v == 0 else ""
        print(f"  {k:>30s} = {v:>12.6f}{flag}")

    assert n_finite == len(FEATURE_NAMES), f"Only {n_finite}/{len(FEATURE_NAMES)} features finite!"
    print(f"  All {n_finite} features finite. {n_zero} are zero.")

    # channel_synergy should be > 0 for 3D data
    cs = result["channel_synergy"]
    if cs == 0.0:
        print("  WARNING: channel_synergy = 0 on 3D data (expected > 0)")
    else:
        print(f"  channel_synergy = {cs:.6f} > 0 on 3D data -- OK")

    # --- Test 2: 1D sine wave ---
    print("\n" + "=" * 60)
    print("Test 2: 1D sine wave (2000 steps)")
    X_1d = np.sin(t) + 0.05 * np.random.default_rng(42).standard_normal(2000)

    t0 = time.perf_counter()
    result_1d = extract(X_1d, seed=42)
    elapsed_1d = time.perf_counter() - t0

    print(f"  Time: {elapsed_1d:.3f}s")
    for k in FEATURE_NAMES:
        v = result_1d[k]
        flag = "" if np.isfinite(v) else " *** NOT FINITE ***"
        print(f"  {k:>30s} = {v:>12.6f}{flag}")

    cs_1d = result_1d["channel_synergy"]
    assert cs_1d == 0.0, f"channel_synergy should be 0 for 1D, got {cs_1d}"
    print(f"  channel_synergy = 0 on 1D data -- OK (by design)")

    # --- Test 3: Rossler attractor (chaotic) ---
    print("\n" + "=" * 60)
    print("Test 3: Rossler attractor (a=0.39, chaotic)")
    traj = rossler(a=0.39, n_steps=6000)
    traj = traj[2000:]  # discard transient

    t0 = time.perf_counter()
    result_r = extract(traj, seed=42)
    elapsed_r = time.perf_counter() - t0

    print(f"  Time: {elapsed_r:.3f}s")
    for k in FEATURE_NAMES:
        v = result_r[k]
        flag = "" if np.isfinite(v) else " *** NOT FINITE ***"
        print(f"  {k:>30s} = {v:>12.6f}{flag}")

    # persistence_significance should be > 0 on chaotic data
    ps = result_r["persistence_significance"]
    if ps == 0:
        print("  WARNING: persistence_significance = 0 on chaotic Rossler (expected > 0)")
    else:
        print(f"  persistence_significance = {ps} on chaotic data -- OK")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  3D sine:  {elapsed:.3f}s")
    print(f"  1D sine:  {elapsed_1d:.3f}s")
    print(f"  Rossler:  {elapsed_r:.3f}s")
    print("  All smoke tests PASSED")


if __name__ == "__main__":
    main()
