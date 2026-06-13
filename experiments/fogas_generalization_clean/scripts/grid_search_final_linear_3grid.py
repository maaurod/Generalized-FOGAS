"""
FinalLinearSolver grid search for the clean 3x3 tabular generalization problem.

The script writes results after every candidate so completed runs survive
interruptions. Use --max-runs for a quick smoke test and --resume for long runs.
"""

import argparse
import itertools
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def find_root(current_path, marker="setup.py"):
    current_path = Path(current_path).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / marker).exists():
            return parent
    return current_path


PROJECT_ROOT = find_root(Path(__file__).resolve())
DATASET_PATH = PROJECT_ROOT / "data" / "datasets_clean" / "3grid.csv"
RESULTS_DIR = PROJECT_ROOT / "data" / "results_clean" / "generalization"
OUTPUT_CSV = RESULTS_DIR / "final_linear_3grid_grid_search.csv"
BEST_CSV = RESULTS_DIR / "final_linear_3grid_grid_search_best.csv"

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_methods.fogas_clean import FOGASEvaluator
from rl_methods.fogas_generalization_clean import (
    FinalLinearSolver,
    LinearFunction,
    LinearQFunction,
    TabularFeatures,
)
from rl_methods.mdp_clean import DiscreteMDP, Planner


SEED = 42
NUM_TRAJECTORIES = 100
MAX_STEPS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALPHA_GRID = [2e-4, 5e-4, 1e-3, 2e-3, 5e-3]
ETA_GRID = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
RHO_GRID = [1e-3, 1e-2, 1e-1, 0.5]
FISHER_DAMPING_GRID = [1e-4, 1e-3, 1e-2]
T_GRID = [1000, 1600, 2500, 4000]
THETA_LR_GRID = [1e-3, 1e-2, 1e-1, 1.0]
D_THETA_GRID = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the 3x3 FinalLinearSolver hyperparameter grid search."
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit the number of candidates. Useful for smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip parameter combinations already present in the output CSV.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes. Use 1 for sequential execution.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch CPU threads per worker. Keep low when using multiple workers.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def configure_worker_threads(torch_threads):
    torch_threads = max(1, int(torch_threads))
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(torch_threads)
    except RuntimeError:
        pass


def build_mdp(device):
    states = torch.arange(9, dtype=torch.long)
    actions = torch.arange(4, dtype=torch.long)

    n_states = len(states)
    n_actions = len(actions)
    gamma = 0.9
    x0 = 0
    goal_grid = 8

    def phi(x, a):
        vec = torch.zeros(n_states * n_actions, dtype=torch.float64, device=device)
        vec[int(x) * n_actions + int(a)] = 1.0
        return vec

    omega = torch.full(
        (n_states * n_actions,),
        -0.1,
        dtype=torch.float64,
        device=device,
    )
    omega[goal_grid * n_actions : goal_grid * n_actions + n_actions] = 1.0

    def to_rc(s):
        return divmod(int(s), 3)

    def to_s(r, c):
        return r * 3 + c

    def next_state(s, a):
        s = int(s)
        a = int(a)

        if s == goal_grid:
            return goal_grid

        r, c = to_rc(s)
        if a == 0:
            r = max(0, r - 1)
        elif a == 1:
            r = min(2, r + 1)
        elif a == 2:
            c = max(0, c - 1)
        elif a == 3:
            c = min(2, c + 1)
        else:
            raise ValueError("Invalid action")

        return to_s(r, c)

    def psi(xp):
        v = torch.zeros(n_states * n_actions, dtype=torch.float64, device=device)
        for x in states:
            for a in actions:
                if next_state(x, a) == int(xp):
                    v[int(x) * n_actions + int(a)] = 1.0
        return v

    mdp = DiscreteMDP(
        states=states,
        actions=actions,
        gamma=gamma,
        x0=x0,
        phi=phi,
        omega=omega,
        psi=psi,
        terminal_states=[goal_grid],
    ).to(device)
    planner = Planner(mdp).to(device)
    return mdp, planner, states, actions, gamma, x0, goal_grid


def candidate_key(row):
    return (
        float(row["alpha"]),
        float(row["eta"]),
        float(row["rho"]),
        float(row["fisher_damping"]),
        int(row["T"]),
        float(row["theta_lr"]),
        float(row["D_theta"]),
    )


def load_existing_results(resume):
    if not resume or not OUTPUT_CSV.exists():
        return [], set()

    df = pd.read_csv(OUTPUT_CSV)
    rows = df.to_dict("records")
    completed = {candidate_key(row) for row in rows}
    return rows, completed


def ordered_results_frame(results):
    df = pd.DataFrame(results)
    if df.empty:
        return df

    return df.sort_values(
        by=[
            "greedy_success_rate",
            "solver_success_rate",
            "greedy_avg_return",
            "solver_avg_return",
            "greedy_v_x0",
            "solver_v_x0",
            "greedy_v_gap",
            "solver_v_gap",
            "elapsed_seconds",
        ],
        ascending=[False, False, False, False, False, False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = ordered_results_frame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    successful = df[df["status"] == "ok"] if not df.empty else df
    if not successful.empty:
        successful.head(1).to_csv(BEST_CSV, index=False)


def evaluate_policy(planner, evaluator, policy_mode, d_star, v_star):
    pi = evaluator.get_policy(policy_mode)
    v_pi, _ = planner.evaluate_policy(pi)
    v_gap = float((d_star * (v_star - v_pi.detach().cpu())).sum().item())
    return {
        f"{policy_mode}_success_rate": float(
            evaluator.success_rate(
                goal_state=8,
                policy_mode=policy_mode,
                num_trajectories=NUM_TRAJECTORIES,
                max_steps=MAX_STEPS,
                seed=SEED,
                terminal_states={8},
            )["policy"]
        ),
        f"{policy_mode}_avg_return": float(
            evaluator.average_return(
                policy_mode=policy_mode,
                num_trajectories=NUM_TRAJECTORIES,
                max_steps=MAX_STEPS,
                seed=SEED,
                terminal_states={8},
            )["policy"]
        ),
        f"{policy_mode}_v_x0": float(v_pi[planner.x0].detach().cpu().item()),
        f"{policy_mode}_v_gap": v_gap,
    }


def blank_metrics():
    return {
        "solver_success_rate": np.nan,
        "solver_avg_return": np.nan,
        "solver_v_x0": np.nan,
        "solver_v_gap": np.nan,
        "greedy_success_rate": np.nan,
        "greedy_avg_return": np.nan,
        "greedy_v_x0": np.nan,
        "greedy_v_gap": np.nan,
        "final_total_loss": np.nan,
        "final_policy_objective": np.nan,
        "final_beta_objective": np.nan,
        "final_q_objective": np.nan,
        "final_theta_norm": np.nan,
        "final_policy_grad_norm": np.nan,
        "final_beta_grad_norm": np.nan,
        "final_theta_grad_norm": np.nan,
    }


def failed_worker_row(params, exc):
    alpha, eta, rho, fisher_damping, T, theta_lr, d_theta = params
    row = {
        "alpha": float(alpha),
        "eta": float(eta),
        "rho": float(rho),
        "fisher_damping": float(fisher_damping),
        "T": int(T),
        "theta_lr": float(theta_lr),
        "D_theta": float(d_theta),
        "theta_mode": "projection",
        "theta_optimizer": "adam",
        "theta_inner_steps": 100,
        "theta_start_mode": "zero",
        "theta_include_beta_cov": False,
        "d_theta_scale": 1.0,
        "beta_update": "fogas_full",
        "policy_optimizer": "npg",
        "policy_gradient": "exact",
        "cg_iters": 50,
        "cg_tol": 1e-10,
        "state_weight_update": "normal",
        "seed": int(SEED),
        "device": str(DEVICE),
        "status": "failed",
        "error": repr(exc),
        "elapsed_seconds": np.nan,
    }
    row.update(blank_metrics())
    return row


def make_solver(n_states, n_actions, gamma, x0, dataset_path, device):
    u_features = TabularFeatures(n_states, n_actions)
    q_features = TabularFeatures(n_states, n_actions)
    policy_features = TabularFeatures(n_states, n_actions)

    return FinalLinearSolver(
        n_states=n_states,
        n_actions=n_actions,
        gamma=gamma,
        x0=x0,
        csv_path=str(dataset_path),
        u_function=LinearFunction(u_features),
        q_function=LinearQFunction(q_features),
        policy_features=policy_features,
        seed=SEED,
        device=device,
        theta_include_beta_cov=False,
        theta_mode="projection",
        theta_optimizer="adam",
        theta_inner_steps=100,
        theta_start_mode="zero",
        d_theta_scale=1.0,
        beta_update="fogas_full",
    )


def run_candidate(params, mdp, planner, dataset_path, device, d_star, v_star):
    alpha, eta, rho, fisher_damping, T, theta_lr, d_theta = params
    start = time.perf_counter()
    row = {
        "alpha": float(alpha),
        "eta": float(eta),
        "rho": float(rho),
        "fisher_damping": float(fisher_damping),
        "T": int(T),
        "theta_lr": float(theta_lr),
        "D_theta": float(d_theta),
        "theta_mode": "projection",
        "theta_optimizer": "adam",
        "theta_inner_steps": 100,
        "theta_start_mode": "zero",
        "theta_include_beta_cov": False,
        "d_theta_scale": 1.0,
        "beta_update": "fogas_full",
        "policy_optimizer": "npg",
        "policy_gradient": "exact",
        "cg_iters": 50,
        "cg_tol": 1e-10,
        "state_weight_update": "normal",
        "seed": int(SEED),
        "device": str(device),
        "status": "ok",
        "error": "",
        "elapsed_seconds": np.nan,
    }
    row.update(blank_metrics())

    try:
        solver = make_solver(
            n_states=mdp.N,
            n_actions=mdp.A,
            gamma=mdp.gamma,
            x0=mdp.x0,
            dataset_path=dataset_path,
            device=device,
        )
        evaluator = FOGASEvaluator(solver=solver, mdp=mdp, planner=planner)

        solver.run(
            alpha=alpha,
            eta=eta,
            rho=rho,
            D_theta=d_theta,
            theta_lr=theta_lr,
            T=T,
            policy_optimizer="npg",
            policy_gradient="exact",
            fisher_damping=fisher_damping,
            cg_iters=50,
            cg_tol=1e-10,
            tqdm_print=False,
            verbose=False,
            state_weight_update="normal",
        )

        row.update(evaluate_policy(planner, evaluator, "solver", d_star, v_star))
        row.update(evaluate_policy(planner, evaluator, "greedy", d_star, v_star))

        diagnostics = solver.get_diagnostics() or []
        if diagnostics:
            final = diagnostics[-1]
            row.update(
                {
                    "final_total_loss": final.get("total_loss", np.nan),
                    "final_policy_objective": final.get("policy_objective", np.nan),
                    "final_beta_objective": final.get("beta_objective", np.nan),
                    "final_q_objective": final.get("q_objective", np.nan),
                    "final_theta_norm": final.get("theta_norm", np.nan),
                    "final_policy_grad_norm": final.get("policy_grad_norm", np.nan),
                    "final_beta_grad_norm": final.get("beta_grad_norm", np.nan),
                    "final_theta_grad_norm": final.get("theta_grad_norm", np.nan),
                }
            )
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = repr(exc)
    finally:
        row["elapsed_seconds"] = float(time.perf_counter() - start)

    return row


def run_candidate_worker(payload):
    params, dataset_path_str, device_str, torch_threads = payload
    configure_worker_threads(torch_threads)
    set_seed(SEED)
    device = torch.device(device_str)
    mdp, planner, _states, _actions, _gamma, _x0, _goal_grid = build_mdp(device)
    d_star = planner.state_mu_star.detach().cpu()
    v_star = planner.v_star.detach().cpu()
    return run_candidate(
        params=params,
        mdp=mdp,
        planner=planner,
        dataset_path=Path(dataset_path_str),
        device=device,
        d_star=d_star,
        v_star=v_star,
    )


def main():
    args = parse_args()
    workers = max(1, int(args.workers))
    torch_threads = max(1, int(args.torch_threads))
    configure_worker_threads(torch_threads)
    set_seed(SEED)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    print(f"Using device: {DEVICE}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Results: {OUTPUT_CSV}")
    print(f"Workers: {workers}")
    print(f"Torch threads per worker: {torch_threads}")

    mdp, planner, states, actions, gamma, x0, goal_grid = build_mdp(DEVICE)
    d_star = planner.state_mu_star.detach().cpu()
    v_star = planner.v_star.detach().cpu()

    all_candidates = list(
        itertools.product(
            ALPHA_GRID,
            ETA_GRID,
            RHO_GRID,
            FISHER_DAMPING_GRID,
            T_GRID,
            THETA_LR_GRID,
            D_THETA_GRID,
        )
    )
    if args.max_runs is not None:
        all_candidates = all_candidates[: max(0, int(args.max_runs))]

    results, completed = load_existing_results(args.resume)
    candidates = [
        candidate
        for candidate in all_candidates
        if candidate_key(
            {
                "alpha": candidate[0],
                "eta": candidate[1],
                "rho": candidate[2],
                "fisher_damping": candidate[3],
                "T": candidate[4],
                "theta_lr": candidate[5],
                "D_theta": candidate[6],
            }
        )
        not in completed
    ]

    print(
        "Total grid size: "
        f"{len(ALPHA_GRID) * len(ETA_GRID) * len(RHO_GRID) * len(FISHER_DAMPING_GRID) * len(T_GRID) * len(THETA_LR_GRID) * len(D_THETA_GRID)}"
    )
    print(f"Candidates to run: {len(candidates)}")
    if args.resume:
        print(f"Resumed rows: {len(results)}")
    if not args.resume and OUTPUT_CSV.exists():
        print("Existing output will be overwritten because --resume was not set.")

    progress = not args.no_progress
    if workers == 1:
        outer = tqdm(candidates, desc="FinalLinearSolver 3-grid search", unit="run", disable=not progress)
        for run_idx, params in enumerate(outer, start=len(results) + 1):
            row = run_candidate(
                params=params,
                mdp=mdp,
                planner=planner,
                dataset_path=DATASET_PATH,
                device=DEVICE,
                d_star=d_star,
                v_star=v_star,
            )
            row["run_idx"] = int(run_idx)
            results.append(row)
            save_results(results)

            if progress:
                outer.set_postfix(
                    {
                        "solver_success": row["solver_success_rate"],
                        "solver_return": row["solver_avg_return"],
                        "status": row["status"],
                    }
                )
    else:
        payloads = [
            (params, str(DATASET_PATH), str(DEVICE), torch_threads)
            for params in candidates
        ]
        next_run_idx = len(results) + 1
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_params = {
                executor.submit(run_candidate_worker, payload): payload[0]
                for payload in payloads
            }
            outer = tqdm(
                as_completed(future_to_params),
                total=len(future_to_params),
                desc="FinalLinearSolver 3-grid search",
                unit="run",
                disable=not progress,
            )
            for future in outer:
                try:
                    row = future.result()
                except Exception as exc:
                    row = failed_worker_row(future_to_params[future], exc)
                row["run_idx"] = int(next_run_idx)
                next_run_idx += 1
                results.append(row)
                save_results(results)

                if progress:
                    outer.set_postfix(
                        {
                            "solver_success": row.get("solver_success_rate", np.nan),
                            "solver_return": row.get("solver_avg_return", np.nan),
                            "status": row.get("status"),
                        }
                    )

    save_results(results)
    df = ordered_results_frame(results)
    ok_count = int((df["status"] == "ok").sum()) if not df.empty else 0
    failed_count = int((df["status"] == "failed").sum()) if not df.empty else 0

    print("\nGrid search complete.")
    print(f"Rows saved: {len(df)}")
    print(f"Successful runs: {ok_count}")
    print(f"Failed runs: {failed_count}")
    print(f"Output CSV: {OUTPUT_CSV}")
    if BEST_CSV.exists():
        print(f"Best row CSV: {BEST_CSV}")
        print("\nTop result:")
        print(pd.read_csv(BEST_CSV).to_string(index=False))


if __name__ == "__main__":
    main()
