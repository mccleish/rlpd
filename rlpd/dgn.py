"""Data-Guided Noise (DGN), zero-mean Gaussian instantiation.

Rollout actions come from N(mu_theta(s), Sigma_phi(s)) where mu_theta is the RL actor's
mean and Sigma_phi(s) = A_phi(s) A_phi(s)^T (cholesky). phi is trained by negative log-likelihood on
prior-data (s, a) pairs with mu_theta held fixed

Choices the paper does not specify (all marked "CHOICE" below):
  - Cholesky diagonal is softplus(raw) + min_diag, to keep it positive.
  - Optimizer learning rate (paper lists AdamW + weight decay 3e-2, but no LR).
  - Dropout is off when sampling rollout noise (on only while fitting).
  - global-norm gradient clipping (grad_clip, default off) as a stability guard. Seems not to affect anything.
"""
from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState


class CholeskyNet(nn.Module):
    """MLP: state -> lower-triangular A_phi(s) with positive diagonal."""
    # choleskynet generates a lower-triangular matrix L such that Sigma = L L^T is the covariance of the Gaussian noise.
    # this allows us to model state-dependent noise in the action space, which is useful for reinforcement learning tasks where the optimal action distribution may vary with the state.
    #modeling state dependent noise means that the noise added to the actions can change depending on the current state of the environment, allowing for more flexible and adaptive exploration strategies.
    # action space in this case is assumed to be continuous and of dimension action_dim, and the network outputs a lower triangular matrix of size (action_dim, action_dim) for each input state.
    action_dim: int
    hidden_dim: int = 256
    num_layers: int = 2
    dropout_rate: float = 0.5
    min_diag: float = 1e-3
    # min_diag is added to the diagonal elements so the matrix is positive definite 
    @nn.compact # nn.compact defines layers in __call__ method
    def __call__(self, x, train: bool = False):
        d = self.action_dim
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x) # nn.Dense creates a fully connected layer with the specified number of units (hidden_dim) and applies it to the input x
            x = nn.relu(x)
            x = nn.Dropout(self.dropout_rate, deterministic=not train)(x)
            # 2 layers: dense, relu, dropout, dense, relu, dropout, dense
        # Output raw lower-triangular entries (including diagonal) of L. The number of entries in a lower triangular matrix of size d is d*(d+1)/2
              # Initialize to zeros to prevent exponential variance compounding
        raw = nn.Dense(
            d * (d + 1) // 2, 
            kernel_init=jax.nn.initializers.zeros
        )(x) #zero initialize to prevent exponential variance compounding
        
        r_idx, c_idx = np.tril_indices(d)
        L = jnp.zeros(raw.shape[:-1] + (d, d), raw.dtype).at[..., r_idx, c_idx].set(raw)
        # L is lower triangular matrix
        
        diag = jax.nn.softplus(jnp.diagonal(L, axis1=-2, axis2=-1)) + self.min_diag
        eye = jnp.eye(d, dtype=raw.dtype)
        return L * (1.0 - eye) + diag[..., :, None] * eye
    # eye is used to create a mask selecting diag elements of L
    # softplus is used to ensure positiveness

def nll_per_sample(L, resid):
    """-log N(resid; 0, L L^T) up to the constant 0.5*d*log(2*pi). L: (B,d,d), resid: (B,d)."""
    z = jax.lax.linalg.triangular_solve(
        L, resid[..., None], left_side=True, lower=True
    )[..., 0]
    logdet = jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)).sum(-1)
    return 0.5 * (z**2).sum(-1) + logdet
     # resid is difference between action taken and mean action predicted by actor network

@jax.jit # jax.jit compiles the function to XLA for faster execution
def _train_step(state, key, obs, mu, act):
    def loss_fn(params):
        L = state.apply_fn({"params": params}, obs, train=True, rngs={"dropout": key})
        return nll_per_sample(L, act - mu).mean()

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss
 # key, obs, mu, act are inputs to the training step, where key is a random key for dropout, obs is the batch of observations, mu is the batch of mean actions predicted by the actor network, and act is the batch of actions taken.
 # function computes the loss and gradients, and updates parameters

def actor_mean(agent, obs, chunk: int = 4096) -> np.ndarray: # actor_mean computes mean action predicted by actor network 
    """Deterministic post-tanh actor action, i.e. mu_theta(s). obs: (d,) or (B, d)."""
    obs = np.asarray(obs, dtype=np.float32)  
    single = obs.ndim == 1 # this checks if the input observation is a single observation (1D array) or a batch of observations (2D array) and reshapes to batch
    if single:
        obs = obs[None] # batch wise processing
    outs = []
    for i in range(0, len(obs), chunk): 
        res = agent.eval_actions(obs[i : i + chunk])
        if isinstance(res, tuple):  # some versions return (actions, new_agent)
            res = res[0] 
        outs.append(np.asarray(res)) 
    out = np.concatenate(outs, 0)
    return out[0] if single else out 

def noise_scale(step: int, tau: float) -> float:
    #Annealing factor exp(-t / tau) from the paper (Adroit: tau = 30000), 0 to disable decay 
    return 1.0 if tau <= 0 else float(np.exp(-step / tau))
 # decay the amount of Gaussian noise added to the actions over time

class DGN:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        seed: int = 0,
        hidden_dim: int = 256,  # paper: 256 Adroit / 128 Robomimic
        num_layers: int = 2,
        dropout_rate: float = 0.5, # WAS 0.5 / 0.1 for fixing bugs
        lr: float = 3e-4,  # CHOICE (not given in the paper)
        weight_decay: float = 3e-2,
        batch_size: int = 128,
        epochs: int = 10,  # paper: 10 Adroit / 2 Robomimic (10 passes/dataset during training), after epoch is done, it counts as one step
        min_diag: float = 1e-3,# WAS  1e-3 for fixing bugs
        grad_clip: float = 0.0,  # WAS stability guard, NOT in the paper (0 = off)
    ):
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = CholeskyNet(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            min_diag=min_diag,
        )
        self._key = jax.random.PRNGKey(seed) # JAX random number generation, i tested with 42 and 43
        self._key, init_key = jax.random.split(self._key)
        params = self.model.init(init_key, jnp.zeros((1, obs_dim), jnp.float32))["params"]
        tx = optax.adamw(lr, weight_decay=weight_decay) 
        if grad_clip > 0:
            tx = optax.chain(optax.clip_by_global_norm(grad_clip), tx)
        self.state = TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)
        self._np_rng = np.random.default_rng(seed)
        self._reported = False
        self._chol = jax.jit(
            lambda p, o: self.model.apply({"params": p}, o, train=False)
        )

    def cholesky(self, obs) -> np.ndarray:
        """A_phi(s) for a batch of states (dropout off). obs: (B, obs_dim)."""
        return np.asarray(self._chol(self.state.params, np.asarray(obs, np.float32)))
        # cholesky function factors observation covariance matrix Sigma into LL^T

    def refit(self, agent, demo_obs: np.ndarray, demo_act: np.ndarray) -> Dict[str, float]: # refit 
    #every dgn_update_interval steps,
    #find the current deterministic mean action for each state
    #find residual between demo_act and mean action
    #run loop using dgn_epochs and dgn_batch_size to update the parameters of the cholesky network to minimize the negative log likelihood of the residuals
    #dgn_epochs is the number of passes through the dataset, dgn_batch_size is the number of samples per batch
    # 
    #Continue training phi (no re-init) on all prior (s, a), mu from the latest actor.
        mu = actor_mean(agent, demo_obs).astype(np.float32) 
        if not self._reported:
            self._report_data(demo_obs, demo_act, mu) # demo_obs is the observations from the demonstration dataset, demo_act is the actions taken in those states, mu is the mean action predicted by the actor network for those states
        n = len(demo_obs) # n is the number of samples in the dataset
        bs = min(self.batch_size, n) 
        epoch_means = []
        for epoch in range(self.epochs):
            perm = self._np_rng.permutation(n) # perm shuffles data for each epoch
            losses = []
            for s in range(0, n - bs + 1, bs):  # drop last partial batch (static shapes)
                idx = perm[s : s + bs] 
                self._key, sub = jax.random.split(self._key) # here we split random key to apply dropout differently for each batch
                self.state, loss = _train_step(
                    self.state, sub, demo_obs[idx], mu[idx], demo_act[idx] ## get the current batch of observations, mean actions, and demonstration actions based on the shuffled indices
                )
                losses.append(loss)
            ep = np.asarray(jnp.stack(losses)) ## stack the losses for each batch in the epoch into a single array
            finite = np.isfinite(ep) 
            if not finite.all():
                bad = int(np.argmin(finite))  # first non-finite batch in this epoch
                raise FloatingPointError(
                    f"DGN covariance fit became non-finite at epoch {epoch}, batch {bad} "
                    f"of {len(ep)}. Losses just before: {ep[max(0, bad - 5):bad].tolist()}. "
                ) # error detection 
            epoch_means.append(float(ep.mean()))  
            if not self._reported: ## debug statement for first refit, can remove 
                print(f"DGN first refit, epoch {epoch}: mean nll {epoch_means[-1]:.4f}")
        self._reported = True

        sel = self._np_rng.choice(n, size=min(2048, n), replace=False) # select random subset of demonstration data
        L = self.cholesky(demo_obs[sel]) # here we compute the Cholesky factor for a random subset of the demonstration observations to evaluate the quality of the fit
        return dict(
            nll_first_epoch=epoch_means[0], 
            nll_last_epoch=epoch_means[-1], #return NLL for first and last epoch to evaluate if the model is learning
            mean_sigma_diag=float(np.diagonal(L, axis1=-2, axis2=-1).mean()), 
            mean_demo_gap=float(np.linalg.norm(demo_act[sel] - mu[sel], axis=-1).mean()), 
            num_prior=float(n), 
        )

    def _report_data(self, obs, act, mu) -> None:
        resid = act - mu
        rs = resid.std(0)
        osd = obs.std(0)
        print("DGN data check (first refit):")
        print(
            f"  finite: obs={np.isfinite(obs).all()} act={np.isfinite(act).all()} "
            f"mu={np.isfinite(mu).all()}"
        )
        print(
            f"  |obs| max {np.abs(obs).max():.4g}; per-dim obs std min/max "
            f"{osd.min():.3g}/{osd.max():.3g}"
        )
        print(
            f"  act range [{act.min():.5f}, {act.max():.5f}]; fraction with |act|>0.9999: "
            f"{(np.abs(act) > 0.9999).mean():.3f}"
        )
        print(f"  per-dim std of (act - mu): min {rs.min():.3g}, max {rs.max():.3g}")

        # report_data is a debug function

    def noise(self, obs) -> np.ndarray:
        L = self.cholesky(np.asarray(obs, np.float32)[None])[0]
        z = self._np_rng.standard_normal(self.action_dim).astype(np.float32)
        return L @ z
        # this function generates a single sample of Gaussian noise with covariance Sigma_phi(s) for a given state obs
        # It first computes the Cholesky factor L of the covariance matrix
        # then samples a standard normal vector z 
        # then computes L @ z to obtain vector sample from the desired Gaussian distribution
        #action_dim is the dimension of output vector that will be added to mean action