"""Fixed RBF search for MultiLinearSBEED on the stochastic 5x5 grid.

This script reuses the common RBF grid-search runner, but evaluates under
stochastic dynamics. The tested configs emphasize entropy, eta, rollout length,
and Fisher damping because those were the most sensitive stability knobs in the
stochastic setting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rbf_grid_search_common import (
    REPO_ROOT,
    add_common_args,
    clear_outputs,
    print_best_result,
    run_fixed_rbf_grid_search,
    summarize_top_results,
    training_kwargs_from_args,
)


# Curated candidates for the noisy-transition grid. Several entries extend the
# default training horizon when a setting needs more data to stabilize.
STOCHASTIC_RBF_30_RUNS = [
    dict(lambda_entropy=0.02, eta=0.01, rollout_length=1, lr_value=1e-3, lr_rho=1e-3, lr_policy=1e-3, batch_size=512, fisher_damping=1e-3, epsilon=0.30),
    dict(lambda_entropy=0.05, eta=0.01, rollout_length=1, lr_value=1e-3, lr_rho=1e-3, lr_policy=1e-3, batch_size=512, fisher_damping=1e-3, epsilon=0.30),
    dict(lambda_entropy=0.05, eta=0.03, rollout_length=1, lr_value=1e-3, lr_rho=1e-3, lr_policy=1e-3, batch_size=512, fisher_damping=1e-3, epsilon=0.35),
    dict(lambda_entropy=0.08, eta=0.03, rollout_length=1, lr_value=1e-3, lr_rho=1e-3, lr_policy=1e-3, batch_size=512, fisher_damping=1e-2, epsilon=0.35),
    dict(lambda_entropy=0.02, eta=0.01, rollout_length=2, lr_value=1e-3, lr_rho=1e-3, lr_policy=1e-3, batch_size=512, fisher_damping=1e-3, epsilon=0.30),
    dict(lambda_entropy=0.02, eta=0.03, rollout_length=2, lr_value=1e-3, lr_rho=1e-3, lr_policy=1e-3, batch_size=512, fisher_damping=1e-3, epsilon=0.30),
    dict(lambda_entropy=0.05, eta=0.03, rollout_length=2, lr_value=1e-3, lr_rho=1e-3, lr_policy=1e-3, batch_size=512, fisher_damping=1e-3, epsilon=0.35),
    dict(lambda_entropy=0.05, eta=0.05, rollout_length=2, lr_value=1e-3, lr_rho=1e-3, lr_policy=1e-3, batch_size=1024, fisher_damping=1e-2, epsilon=0.35),
    dict(lambda_entropy=0.08, eta=0.03, rollout_length=2, lr_value=1e-3, lr_rho=1e-3, lr_policy=1e-3, batch_size=1024, fisher_damping=1e-2, epsilon=0.40),
    dict(lambda_entropy=0.05, eta=0.01, rollout_length=2, lr_value=1e-3, lr_rho=1e-3, lr_policy=3e-3, batch_size=512, fisher_damping=1e-2, epsilon=0.30),
    dict(lambda_entropy=0.05, eta=0.01, rollout_length=3, lr_value=1e-3, lr_rho=3e-4, lr_policy=1e-3, batch_size=1024, fisher_damping=1e-2, epsilon=0.40),
    dict(lambda_entropy=0.05, eta=0.03, rollout_length=3, lr_value=1e-3, lr_rho=3e-4, lr_policy=1e-3, batch_size=1024, fisher_damping=1e-2, epsilon=0.40),
    dict(lambda_entropy=0.08, eta=0.03, rollout_length=3, lr_value=3e-4, lr_rho=3e-4, lr_policy=1e-3, batch_size=1024, fisher_damping=1e-2, epsilon=0.40),
    dict(lambda_entropy=0.08, eta=0.05, rollout_length=3, lr_value=3e-4, lr_rho=3e-4, lr_policy=3e-4, batch_size=1024, fisher_damping=1e-2, epsilon=0.40),
    dict(lambda_entropy=0.02, eta=0.03, rollout_length=1, lr_value=1e-3, lr_rho=1e-3, lr_policy=1e-3, batch_size=1024, fisher_damping=1e-2, epsilon=0.30),
    dict(lambda_entropy=0.05, eta=0.03, rollout_length=1, lr_value=1e-3, lr_rho=1e-3, lr_policy=3e-3, batch_size=1024, fisher_damping=1e-2, epsilon=0.35),
    dict(lambda_entropy=0.02, eta=0.05, rollout_length=2, lr_value=1e-3, lr_rho=3e-4, lr_policy=1e-3, batch_size=1024, fisher_damping=1e-2, epsilon=0.40),
    dict(lambda_entropy=0.05, eta=0.10, rollout_length=2, lr_value=1e-3, lr_rho=3e-4, lr_policy=1e-3, batch_size=1024, fisher_damping=1e-2, epsilon=0.40),
    dict(lambda_entropy=0.02, eta=0.03, rollout_length=2, lr_value=3e-4, lr_rho=3e-4, lr_policy=1e-3, batch_size=512, fisher_damping=1e-3, epsilon=0.35),
    dict(lambda_entropy=0.05, eta=0.03, rollout_length=2, lr_value=3e-4, lr_rho=3e-4, lr_policy=1e-3, batch_size=512, fisher_damping=1e-2, epsilon=0.35),
    dict(lambda_entropy=0.08, eta=0.03, rollout_length=2, lr_value=3e-4, lr_rho=3e-4, lr_policy=3e-4, batch_size=1024, fisher_damping=1e-2, epsilon=0.40),
    dict(lambda_entropy=0.05, eta=0.01, rollout_length=1, lr_value=3e-4, lr_rho=3e-4, lr_policy=3e-4, batch_size=512, fisher_damping=1e-3, epsilon=0.30),
    dict(lambda_entropy=0.02, eta=0.0, rollout_length=1, lr_value=1e-3, lr_rho=1e-3, lr_policy=1e-3, batch_size=512, fisher_damping=1e-3, epsilon=0.30),
    dict(lambda_entropy=0.05, eta=0.0, rollout_length=1, lr_value=1e-3, lr_rho=1e-3, lr_policy=1e-3, batch_size=512, fisher_damping=1e-3, epsilon=0.35),
    dict(lambda_entropy=0.05, eta=0.0, rollout_length=2, lr_value=1e-3, lr_rho=1e-3, lr_policy=1e-3, batch_size=1024, fisher_damping=1e-2, epsilon=0.40),
    dict(lambda_entropy=0.01, eta=0.03, rollout_length=1, lr_value=1e-3, lr_rho=1e-3, lr_policy=3e-3, batch_size=512, fisher_damping=1e-3, epsilon=0.30),
    dict(lambda_entropy=0.01, eta=0.03, rollout_length=2, lr_value=1e-3, lr_rho=1e-3, lr_policy=1e-3, batch_size=512, fisher_damping=1e-3, epsilon=0.35),
    dict(lambda_entropy=0.02, eta=0.10, rollout_length=2, lr_value=1e-3, lr_rho=3e-4, lr_policy=1e-3, batch_size=1024, fisher_damping=1e-2, epsilon=0.40),
    dict(lambda_entropy=0.08, eta=0.10, rollout_length=1, lr_value=1e-3, lr_rho=3e-4, lr_policy=1e-3, batch_size=1024, fisher_damping=1e-2, epsilon=0.40),
    dict(lambda_entropy=0.05, eta=0.03, rollout_length=3, lr_value=1e-3, lr_rho=1e-3, lr_policy=3e-4, batch_size=1024, fisher_damping=3e-2, epsilon=0.50),
    dict(lambda_entropy=0.01, eta=0.03, rollout_length=1, lr_value=1e-3, lr_rho=1e-3, lr_policy=3e-3, batch_size=512, fisher_damping=1e-3, epsilon=0.30, episodes=2200, collect_per_episode=25, updates_per_episode=15, initial_collect_steps=2500),
    dict(lambda_entropy=0.01, eta=0.03, rollout_length=2, lr_value=1e-3, lr_rho=1e-3, lr_policy=1e-3, batch_size=512, fisher_damping=1e-3, epsilon=0.35, episodes=2200, collect_per_episode=25, updates_per_episode=15, initial_collect_steps=2500),
    dict(lambda_entropy=0.05, eta=0.01, rollout_length=3, lr_value=1e-3, lr_rho=3e-4, lr_policy=1e-3, batch_size=1024, fisher_damping=1e-2, epsilon=0.40, episodes=2200, collect_per_episode=25, updates_per_episode=15, initial_collect_steps=2500),
]

# Defaults shared by most candidates unless a config dictionary overrides them.
STOCHASTIC_RBF_TRAINING_KWARGS = dict(
    episodes=1500,
    collect_per_episode=20,
    updates_per_episode=10,
    initial_collect_steps=1500,
    max_buffer_size=12000,
    tau=100000.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed MultiLinearSBEED RBF configs on the stochastic 5x5 gridworld."
    )
    add_common_args(
        parser,
        default_output_dir=REPO_ROOT / "data/results/sbeed",
        default_training_kwargs=STOCHASTIC_RBF_TRAINING_KWARGS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if args.overwrite:
        clear_outputs(output_dir)

    torch.set_num_threads(args.torch_threads)
    results, best = run_fixed_rbf_grid_search(
        name="stochastic_rbf",
        configs=STOCHASTIC_RBF_30_RUNS,
        stochastic=True,
        training_kwargs=training_kwargs_from_args(args),
        output_dir=output_dir,
        device=torch.device(args.device),
        base_seed=args.base_seed,
        n_runs=args.n_runs,
        eval_every_episodes=args.eval_every_episodes,
        n_eval_episodes_during=args.n_eval_episodes_during,
        n_eval_episodes_final=args.n_eval_episodes_final,
        max_steps_per_eval_episode=args.max_steps_per_eval_episode,
        early_stop_after_episodes=args.early_stop_after_episodes,
        early_stop_margin=None if args.disable_early_stop else args.early_stop_margin,
        workers=args.workers,
        resume=args.resume,
    )
    summarize_top_results(results, top_k=10)
    print_best_result(best)


if __name__ == "__main__":
    main()
