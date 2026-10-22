"""Sanity checks for rlpd/dgn.py on synthetic data. No RL, no environment, runs on CPU.

Run from the repo root:  python dgn_selftest.py
Thresholds are deliberately loose; read the printed numbers, not just PASS/FAIL.
"""
import sys

import numpy as np

from rlpd.dgn import DGN, noise_scale

OBS_DIM, ACT_DIM = 8, 4
rng = np.random.default_rng(0)


class FakeAgent:
    """Stands in for the RLPD agent: eval_actions returns tanh(obs @ W) like tanh(mean)."""

    def __init__(self, W):
        self.W = W

    def eval_actions(self, obs):
        return np.tanh(np.asarray(obs) @ self.W)


def make_data(n, state_dependent):
    obs = rng.standard_normal((n, OBS_DIM)).astype(np.float32)
    W = (rng.standard_normal((OBS_DIM, ACT_DIM)) * 0.3).astype(np.float32)
    A = np.tril(rng.uniform(-0.3, 0.3, (ACT_DIM, ACT_DIM)))
    A[np.diag_indices(ACT_DIM)] = [0.4, 0.3, 0.2, 0.1]
    scale = np.where(obs[:, 0] > 0, 0.5, 0.1) if state_dependent else np.ones(n)
    z = rng.standard_normal((n, ACT_DIM))
    act = np.tanh(obs @ W) + scale[:, None] * (z @ A.T)
    return obs, act.astype(np.float32), FakeAgent(W), A


def fit(obs, act, agent):
    dgn = DGN(OBS_DIM, ACT_DIM, seed=0, hidden_dim=64, dropout_rate=0.1,
              lr=1e-3, batch_size=128, epochs=40)
    info = dgn.refit(agent, obs, act)
    print(f"  nll first epoch {info['nll_first_epoch']:.3f} -> last {info['nll_last_epoch']:.3f}")
    return dgn, info


ok = True


def check(name, cond, detail):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")


# 1. constant covariance is recovered
print("Test 1: constant covariance")
obs, act, agent, A = make_data(20000, state_dependent=False)
dgn, info = fit(obs, act, agent)
L = dgn.cholesky(obs[:2000])
sigma_hat = np.mean(L @ L.transpose(0, 2, 1), axis=0)
sigma_true = A @ A.T
rel = np.linalg.norm(sigma_hat - sigma_true) / np.linalg.norm(sigma_true)
check("covariance recovery", rel < 0.3, f"relative Frobenius error {rel:.3f}")
check("loss decreased", info["nll_last_epoch"] < info["nll_first_epoch"], "")

# 2. state dependence is picked up (true scale ratio is 5x between obs[0] > 0 and < 0)
print("Test 2: state-dependent scale")
obs, act, agent, A = make_data(20000, state_dependent=True)
dgn, _ = fit(obs, act, agent)
L = dgn.cholesky(obs[:4000])
d = np.diagonal(L, axis1=-2, axis2=-1).mean(-1)
ratio = d[obs[:4000, 0] > 0].mean() / d[obs[:4000, 0] <= 0].mean()
check("state-dependent scale", 3.0 < ratio < 7.0, f"learned scale ratio {ratio:.2f} (true 5.0)")

# 3. sampler matches the learned covariance
print("Test 3: sampler")
s0 = obs[0]
Ls = dgn.cholesky(s0[None])[0]
samples = np.stack([dgn.noise(s0) for _ in range(20000)])
emp = np.cov(samples.T)
rel = np.linalg.norm(emp - Ls @ Ls.T) / np.linalg.norm(Ls @ Ls.T)
check("noise() covariance", rel < 0.1, f"relative error {rel:.3f}")

# 4. annealing schedule
check("noise_scale", abs(noise_scale(0, 30000) - 1) < 1e-9
      and abs(noise_scale(30000, 30000) - np.exp(-1)) < 1e-9
      and noise_scale(12345, 0) == 1.0, "exp(-t/tau), tau<=0 disables")

print("ALL PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
