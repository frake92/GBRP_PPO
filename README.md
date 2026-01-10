# SB3 PPO MLP (per-update CSV)

This repo includes a minimal, clean training script using Stable-Baselines3 PPO with an **MLP policy** and **CSV metrics written after every policy update**.

## Setup

```bash
cd /home/rake
source ML-venv/bin/activate
cd /home/rake/Research3
pip install -r requirements.txt
```

## Train

Default behavior:
- PPO policy update happens every `n_steps=500` rollout steps (per env)
- Runs `policy_updates=500` updates total
- Writes metrics to `runs/<run-name>/progress.csv` (one row per update)

```bash
cd /home/rake
source ML-venv/bin/activate
cd /home/rake/Research3
python network.py
```

Override key knobs:

```bash
cd /home/rake
source ML-venv/bin/activate
cd /home/rake/Research3
python network.py --env CartPole-v1 --policy-updates 50 --n-steps 500 --n-envs 1
```

Enable GBRP (reward-protection layer):

```bash
python network.py --env CartPole-v1 --gbrp
```

Outputs:
- `runs/<run-name>/progress.csv`
- `runs/<run-name>/final_model.zip`
- TensorBoard logs under `runs/<run-name>/tb/`

## Plot metrics

This reads SB3's `progress.csv` and writes one PNG per metric.

```bash
cd /home/rake
source ML-venv/bin/activate
cd /home/rake/Research3
python plot_metrics.py --csv runs/_smoketest/progress.csv --out plots/_smoketest
```

Pick specific metrics:

```bash
cd /home/rake
source ML-venv/bin/activate
cd /home/rake/Research3
python plot_metrics.py --csv runs/<run-name>/progress.csv --keys rollout/ep_rew_mean,train/value_loss --rolling 5
```
