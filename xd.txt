from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch as th
except Exception as e:  # pragma: no cover
    th = None  # type: ignore[assignment]

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


@dataclass
class GBRPConfig:
    group_size: int = 5
    baseline_window: int = 3

    n1: int = 1
    n2: int = 3

    tau_a: float = 0.6
    tau_e: float = 0.15
    tau_r: float = 0.2
    tau_v: float = 1.5
    tau_spike: float = 1.5

    tau_H: float = 0.05

    eps: float = 1e-8


@dataclass
class _UpdateStats:
    raw_reward_mean_step: float
    ep_len_mean: float

    action_imbalance: float

    entropy_mean: float
    entropy_std: float

    value_loss_proxy: float
    explained_variance_raw: float


@dataclass
class _GroupSummary:
    barR_raw: float
    barL: float
    H_mean: float

    action_imbalance: float
    entropy_oscillation: float

    value_loss_var: float
    explained_variance_raw: float


class GBRPVecEnvWrapper(VecEnvWrapper):
    """Applies the current GBRP reward multiplier (lambda) *during collection*.

    Detection must use raw rewards, so this wrapper also stores per-step raw rewards
    for the callback to read at rollout end.
    """

    def __init__(self, venv: VecEnv, *, initial_lambda: float = 1.0):
        super().__init__(venv)
        self._lambda: float = float(initial_lambda)

        self._action_space_discrete_n: Optional[int] = None
        try:
            # VecEnv has a single action_space describing all envs.
            from gymnasium import spaces

            if isinstance(self.action_space, spaces.Discrete):
                self._action_space_discrete_n = int(self.action_space.n)
        except Exception:
            self._action_space_discrete_n = None

        self._pending_actions: Optional[np.ndarray] = None
        self.reset_rollout_tracking()

    @property
    def current_lambda(self) -> float:
        return self._lambda

    def set_lambda(self, new_lambda: float) -> None:
        self._lambda = float(np.clip(new_lambda, 0.0, 1.0))

    def reset_rollout_tracking(self) -> None:
        self._raw_rewards_steps: List[np.ndarray] = []

        self._action_counts: Optional[np.ndarray]
        if self._action_space_discrete_n is not None:
            self._action_counts = np.zeros(self._action_space_discrete_n, dtype=np.int64)
        else:
            self._action_counts = None

        self._action_total: int = 0

        self._raw_ep_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._raw_ep_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._completed_raw_ep_returns: List[float] = []
        self._completed_raw_ep_lengths: List[int] = []

    def pop_rollout_tracking(self) -> Dict[str, Any]:
        raw_rewards = np.asarray(self._raw_rewards_steps, dtype=np.float64)  # (T, n_envs)
        if raw_rewards.ndim != 2:
            # Be defensive if something odd happens.
            raw_rewards = raw_rewards.reshape((-1, self.num_envs))

        action_imbalance = float("nan")
        if self._action_counts is not None and self._action_total > 0:
            action_imbalance = float(self._action_counts.max() / max(1, self._action_total))

        ep_len_mean = float("nan")
        if self._completed_raw_ep_lengths:
            ep_len_mean = float(np.mean(self._completed_raw_ep_lengths))

        return {
            "raw_rewards": raw_rewards,
            "action_imbalance": action_imbalance,
            "ep_len_mean": ep_len_mean,
            "completed_raw_ep_returns": list(self._completed_raw_ep_returns),
            "completed_raw_ep_lengths": list(self._completed_raw_ep_lengths),
            "lambda": self._lambda,
        }

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = np.array(actions)

        if self._action_counts is not None:
            flat = np.array(actions).reshape(-1)
            # Some policies output shape (n_envs, 1)
            flat = flat.astype(np.int64)
            for a in flat:
                if 0 <= int(a) < len(self._action_counts):
                    self._action_counts[int(a)] += 1
                    self._action_total += 1

        return self.venv.step_async(actions)

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        obs, rewards_raw, dones, infos = self.venv.step_wait()

        rewards_raw = np.asarray(rewards_raw, dtype=np.float64)
        self._raw_rewards_steps.append(rewards_raw.copy())

        # Track raw episodic stats.
        self._raw_ep_returns += rewards_raw
        self._raw_ep_lengths += 1
        for i, done in enumerate(dones):
            if bool(done):
                self._completed_raw_ep_returns.append(float(self._raw_ep_returns[i]))
                self._completed_raw_ep_lengths.append(int(self._raw_ep_lengths[i]))
                self._raw_ep_returns[i] = 0.0
                self._raw_ep_lengths[i] = 0

        lam = self._lambda
        rewards_mod = rewards_raw * lam

        # Attach diagnostics (raw reward is needed for anti self-deception).
        for i in range(len(infos)):
            infos[i] = dict(infos[i])
            infos[i]["gbrp/raw_reward"] = float(rewards_raw[i])
            infos[i]["gbrp/lambda"] = float(lam)

        return obs, rewards_mod, dones, infos

    def reset(self) -> np.ndarray:
        """Reset the underlying VecEnv.

        SB3's VecEnv interface requires wrappers to implement `reset()`.
        Some versions return only `obs`, while others may return `(obs, info)`.
        """

        self._pending_actions = None
        # Reset per-rollout tracking so the next rollout starts clean.
        self.reset_rollout_tracking()

        out = self.venv.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, _info = out
            return obs
        return out


def _find_gbrp_wrapper(env: VecEnv) -> Optional[GBRPVecEnvWrapper]:
    cur: Any = env
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, GBRPVecEnvWrapper):
            return cur
        cur = getattr(cur, "venv", None)
    return None


class GBRPCallback(BaseCallback):
    """Implements the abstract.tex GBRP (Group-Based Reward Protection) logic.

    - Detects reward hacking signals ONLY from raw rewards + raw-derived metrics.
    - Applies forward-looking penalty by changing the env wrapper lambda for the next group.
    """

    def __init__(self, cfg: GBRPConfig | None = None, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.cfg = cfg or GBRPConfig()

        self._wrapper: Optional[GBRPVecEnvWrapper] = None

        self._updates_in_group: List[_UpdateStats] = []
        self._groups: List[_GroupSummary] = []
        self._value_loss_var_history: List[float] = []

        self._pending_next_lambda: float = 1.0

        self._last_group_entropy_mean: Optional[float] = None

    def _on_training_start(self) -> None:
        env = self.model.get_env()
        self._wrapper = _find_gbrp_wrapper(env)
        if self._wrapper is None:
            raise RuntimeError(
                "GBRPCallback requires the environment to be wrapped with GBRPVecEnvWrapper."
            )

        self._pending_next_lambda = float(self._wrapper.current_lambda)

    def _on_rollout_start(self) -> None:
        if self._wrapper is not None:
            self._wrapper.reset_rollout_tracking()

    def _compute_entropy_stats(self) -> Tuple[float, float]:
        if th is None:
            return float("nan"), float("nan")

        rb = self.model.rollout_buffer
        obs = rb.observations
        actions = rb.actions

        # RolloutBuffer stores (n_steps, n_envs, ...) arrays.
        obs_t = th.as_tensor(obs).to(self.model.device)
        actions_t = th.as_tensor(actions).to(self.model.device)

        obs_t = obs_t.reshape((-1,) + obs_t.shape[2:])
        actions_t = actions_t.reshape((-1,) + actions_t.shape[2:])

        with th.no_grad():
            _, _, entropy = self.model.policy.evaluate_actions(obs_t, actions_t)

        if entropy is None:
            return float("nan"), float("nan")

        ent = entropy.detach().cpu().numpy().astype(np.float64)
        return float(np.mean(ent)), float(np.std(ent))

    def _compute_raw_returns(self, raw_rewards: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute bootstrapped discounted returns from raw rewards.

        Returns (returns_raw, dones_t) with shape (T, n_envs).
        """

        rb = self.model.rollout_buffer
        T, n_envs = raw_rewards.shape

        values = np.asarray(rb.values, dtype=np.float64)
        values = values.reshape((T, n_envs))

        # done at time t is episode_starts[t+1] (SB3 convention)
        episode_starts = np.asarray(rb.episode_starts, dtype=bool).reshape((T, n_envs))
        dones_t = np.zeros((T, n_envs), dtype=bool)
        if T > 1:
            dones_t[:-1] = episode_starts[1:]

        last_episode_starts = getattr(self.model, "_last_episode_starts", None)
        if last_episode_starts is not None:
            dones_t[-1] = np.asarray(last_episode_starts, dtype=bool).reshape((n_envs,))

        # Bootstrap with V(s_T)
        last_obs = getattr(self.model, "_last_obs", None)
        if th is None or last_obs is None:
            last_values = values[-1]
        else:
            with th.no_grad():
                last_values_t = self.model.policy.predict_values(th.as_tensor(last_obs).to(self.model.device))
            last_values = last_values_t.detach().cpu().numpy().reshape((n_envs,)).astype(np.float64)

        gamma = float(getattr(self.model, "gamma", 0.99))

        returns = np.zeros_like(values)
        next_return = last_values
        for t in reversed(range(T)):
            done = dones_t[t]
            next_return = raw_rewards[t] + gamma * (1.0 - done.astype(np.float64)) * next_return
            returns[t] = next_return
            # If episode ended at t, the return recursion already reset via (1-done).

        return returns, dones_t

    def _on_rollout_end(self) -> None:
        assert self._wrapper is not None

        tracking = self._wrapper.pop_rollout_tracking()
        raw_rewards: np.ndarray = tracking["raw_rewards"]
        action_imbalance: float = float(tracking["action_imbalance"])
        ep_len_mean: float = float(tracking["ep_len_mean"])

        # Raw reward mean (per step) for this update.
        raw_reward_mean_step = float(np.mean(raw_rewards)) if raw_rewards.size else float("nan")

        entropy_mean, entropy_std = self._compute_entropy_stats()

        returns_raw, dones_t = self._compute_raw_returns(raw_rewards)

        rb = self.model.rollout_buffer
        values = np.asarray(rb.values, dtype=np.float64).reshape(returns_raw.shape)

        # TD(0) errors as a value-loss proxy (raw).
        gamma = float(getattr(self.model, "gamma", 0.99))
        values_next = np.zeros_like(values)
        values_next[:-1] = values[1:]

        # Bootstrap last step with V(s_T)
        last_obs = getattr(self.model, "_last_obs", None)
        if th is None or last_obs is None:
            values_next[-1] = values[-1]
        else:
            with th.no_grad():
                last_values_t = self.model.policy.predict_values(th.as_tensor(last_obs).to(self.model.device))
            values_next[-1] = last_values_t.detach().cpu().numpy().reshape((values.shape[1],)).astype(np.float64)

        td = raw_rewards + gamma * (1.0 - dones_t.astype(np.float64)) * values_next - values
        value_loss_proxy = float(np.mean(td * td))

        ev_raw = float(explained_variance(returns_raw.flatten(), values.flatten()))

        self._updates_in_group.append(
            _UpdateStats(
                raw_reward_mean_step=raw_reward_mean_step,
                ep_len_mean=ep_len_mean,
                action_imbalance=action_imbalance,
                entropy_mean=entropy_mean,
                entropy_std=entropy_std,
                value_loss_proxy=value_loss_proxy,
                explained_variance_raw=ev_raw,
            )
        )

        # Log per-update diagnostics (raw vs modified tracked separately).
        self.logger.record("rewards/raw_mean_step", raw_reward_mean_step)
        self.logger.record("rewards/modified_mean_step", raw_reward_mean_step * float(self._wrapper.current_lambda))
        self.logger.record("gbrp/current_lambda", float(self._wrapper.current_lambda))

        if len(self._updates_in_group) < self.cfg.group_size:
            return

        # ----- Group end: compute signals on RAW-only metrics -----
        updates = self._updates_in_group
        self._updates_in_group = []

        barR_raw = float(np.mean([u.raw_reward_mean_step for u in updates]))

        ep_lens = [u.ep_len_mean for u in updates if np.isfinite(u.ep_len_mean)]
        barL = float(np.mean(ep_lens)) if ep_lens else float("nan")

        H_mean = float(np.nanmean([u.entropy_mean for u in updates]))
        entropy_series = np.asarray([u.entropy_mean for u in updates], dtype=np.float64)
        entropy_osc = float(np.nanstd(entropy_series))

        action_imb = float(np.nanmax([u.action_imbalance for u in updates]))

        value_loss_series = np.asarray([u.value_loss_proxy for u in updates], dtype=np.float64)
        value_loss_var = float(np.var(value_loss_series))

        ev_group = float(np.nanmean([u.explained_variance_raw for u in updates]))

        group = _GroupSummary(
            barR_raw=barR_raw,
            barL=barL,
            H_mean=H_mean,
            action_imbalance=action_imb,
            entropy_oscillation=entropy_osc,
            value_loss_var=value_loss_var,
            explained_variance_raw=ev_group,
        )
        self._groups.append(group)
        self._value_loss_var_history.append(value_loss_var)

        # Rolling baselines from previous m groups (raw).
        m = int(self.cfg.baseline_window)
        eps = float(self.cfg.eps)
        prev_groups = self._groups[:-1]

        mu_R = float("nan")
        mu_L = float("nan")
        if len(prev_groups) >= m:
            window = prev_groups[-m:]
            mu_R = float(np.mean([g.barR_raw for g in window]))
            mu_L = float(np.nanmean([g.barL for g in window]))

        # L1 signals
        a_imb = action_imb

        e_drop = float("nan")
        if self._last_group_entropy_mean is not None and np.isfinite(self._last_group_entropy_mean) and np.isfinite(H_mean):
            e_drop = float((self._last_group_entropy_mean - H_mean) / (abs(self._last_group_entropy_mean) + eps))

        r_grad = float("nan")
        if np.isfinite(mu_R) and np.isfinite(barR_raw):
            r_grad = float((barR_raw - mu_R) / (abs(mu_R) + eps))

        v_inst = float("nan")
        if len(self._value_loss_var_history) >= 2:
            prev_var = float(self._value_loss_var_history[-2])
            v_inst = float(value_loss_var / (prev_var + eps))

        S1 = 0
        if np.isfinite(a_imb) and a_imb > self.cfg.tau_a:
            S1 += 1
        if np.isfinite(e_drop) and e_drop > self.cfg.tau_e:
            S1 += 1
        if np.isfinite(r_grad) and r_grad > self.cfg.tau_r:
            S1 += 1
        if np.isfinite(v_inst) and v_inst > self.cfg.tau_v:
            S1 += 1

        level1 = int(S1 >= int(self.cfg.n1))

        # L2 signals
        r_spike = float("nan")
        if np.isfinite(mu_R) and np.isfinite(barR_raw):
            r_spike = float(barR_raw / (mu_R + eps))

        E_len = 0
        if np.isfinite(mu_L) and np.isfinite(mu_R) and np.isfinite(barL) and np.isfinite(barR_raw):
            E_len = int((barL > mu_L) and (barR_raw > mu_R))

        E_EV = 0
        if np.isfinite(ev_group):
            E_EV = int(0.3 < ev_group < 0.7)

        E_ent = 0
        if np.isfinite(entropy_osc):
            E_ent = int(entropy_osc > self.cfg.tau_H)

        E_v = int(np.isfinite(v_inst) and (v_inst > self.cfg.tau_v))

        S2 = 0
        if np.isfinite(r_spike) and (r_spike > self.cfg.tau_spike):
            S2 += 1
        S2 += E_len + E_EV + E_ent + E_v

        level2 = int(S2 >= int(self.cfg.n2))

        # Penalty selection (forward-looking)
        if level2:
            next_lambda = 0.4
        elif level1 and (np.isfinite(r_grad) and (r_grad > 0.25)):
            next_lambda = 0.7
        elif level1:
            next_lambda = 0.9
        else:
            next_lambda = 1.0

        # Apply to next group collection.
        self._wrapper.set_lambda(next_lambda)
        self._pending_next_lambda = next_lambda

        # Remember entropy baseline for the next group.
        self._last_group_entropy_mean = H_mean

        # Log GBRP group-level metrics.
        self.logger.record("gbrp/level1_active", float(level1))
        self.logger.record("gbrp/level2_active", float(level2))
        self.logger.record("gbrp/S1_count", float(S1))
        self.logger.record("gbrp/S2_count", float(S2))
        self.logger.record("gbrp/reward_grad", float(r_grad if np.isfinite(r_grad) else 0.0))
        self.logger.record("gbrp/reward_spike", float(r_spike if np.isfinite(r_spike) else 0.0))
        self.logger.record("gbrp/episode_len_reward_coupling", float(E_len))
        self.logger.record("gbrp/value_loss_instability", float(v_inst if np.isfinite(v_inst) else 0.0))
        self.logger.record("gbrp/explained_variance", float(ev_group if np.isfinite(ev_group) else 0.0))
        self.logger.record("gbrp/entropy_oscillation", float(entropy_osc if np.isfinite(entropy_osc) else 0.0))

    def _on_step(self) -> bool:
        return True
