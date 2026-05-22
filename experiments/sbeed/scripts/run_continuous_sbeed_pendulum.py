"""Small smoke run for ContinuousSBEED on Pendulum-v1.

Use this script to verify that the continuous solver, Gymnasium environment,
and parametrization modules run end-to-end before launching the larger grid
search. Defaults are intentionally short and are not meant to be the best
reported hyperparameters.
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
import torch

from rl_methods.sbeed.features import (
    ContinuousNeuralRhoParam,
    ContinuousNeuralValueParam,
    ContinuousStateActionMLPModule,
    ContinuousStateMLPValueModule,
    RFFGaussianPolicyParam,
)
from rl_methods.sbeed.solvers import ContinuousSBEED


def build_solver(env, args: argparse.Namespace) -> ContinuousSBEED:
    """Create the value, rho, and Gaussian policy modules used by the solver."""
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    value_param = ContinuousNeuralValueParam(
        ContinuousStateMLPValueModule(
            obs_dim=obs_dim,
            hidden_sizes=(args.hidden_size, args.hidden_size),
            dtype=torch.float32,
        )
    )
    rho_param = ContinuousNeuralRhoParam(
        ContinuousStateActionMLPModule(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=(args.hidden_size, args.hidden_size),
            output_dim=1,
            dtype=torch.float32,
        )
    )
    policy_param = RFFGaussianPolicyParam(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_features=args.rff_features,
        nu=args.nu,
        init_log_std=args.init_log_std,
        dtype=torch.float32,
        seed=args.seed,
    )
    return ContinuousSBEED(
        obs_dim=obs_dim,
        action_dim=action_dim,
        gamma=args.gamma,
        value_param=value_param,
        rho_param=rho_param,
        policy_param=policy_param,
        lambda_entropy=args.lambda_entropy,
        eta=args.eta,
        lr_value=args.lr_value,
        lr_rho=args.lr_rho,
        lr_policy=args.lr_policy,
        batch_size=args.batch_size,
        rollout_length=args.rollout_length,
        max_buffer_size=args.max_buffer_size,
        fisher_damping=args.fisher_damping,
        cg_iters=args.cg_iters,
        tau=args.tau,
        seed=args.seed,
        device=args.device,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-run ContinuousSBEED on Pendulum-v1.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--initial-random-steps", type=int, default=256)
    parser.add_argument("--collect-per-episode", type=int, default=128)
    parser.add_argument("--updates-per-episode", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rollout-length", type=int, default=1)
    parser.add_argument("--max-buffer-size", type=int, default=5000)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--rff-features", type=int, default=100)
    parser.add_argument("--nu", type=float, default=None)
    parser.add_argument("--init-log-std", type=float, default=-0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda-entropy", type=float, default=0.01)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--lr-value", type=float, default=1e-3)
    parser.add_argument("--lr-rho", type=float, default=1e-3)
    parser.add_argument("--lr-policy", type=float, default=1e-3)
    parser.add_argument("--fisher-damping", type=float, default=1e-2)
    parser.add_argument("--cg-iters", type=int, default=10)
    parser.add_argument("--tau", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = gym.make("Pendulum-v1")
    solver = build_solver(env, args)
    result = solver.run_env(
        env,
        episodes=args.episodes,
        initial_random_steps=args.initial_random_steps,
        collect_per_episode=args.collect_per_episode,
        updates_per_episode=args.updates_per_episode,
        log_every=args.log_every,
    )
    returns = result["episode_returns"]
    avg_return = float(np.mean(returns[-10:])) if returns else float("nan")
    print(f"buffer_size={result['buffer_size']} recent_avg_return={avg_return:.3f}")
    env.close()


if __name__ == "__main__":
    main()
