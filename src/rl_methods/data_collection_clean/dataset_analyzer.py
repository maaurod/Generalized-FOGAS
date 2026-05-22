"""
DatasetAnalyzer
---------------

Small, data-only analyzer for discrete offline RL datasets.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


class DatasetAnalyzer:
    """
    Analyze state/action frequencies and feature coverage for a dataset.

    Parameters
    ----------
    data : str, pathlib.Path, or pandas.DataFrame
        CSV path or DataFrame with at least ``state`` and ``action`` columns.
    """

    REQUIRED_COLUMNS = ("state", "action")

    def __init__(self, data: Union[str, Path, pd.DataFrame]):
        if isinstance(data, pd.DataFrame):
            self.source = "dataframe"
            self.df = data.copy()
        else:
            self.source = str(data)
            self.df = pd.read_csv(data)

        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        if missing:
            raise ValueError(f"Dataset is missing required column(s): {missing}")

        self.df["state"] = self.df["state"].astype(int)
        self.df["action"] = self.df["action"].astype(int)

        self._state_counts = self.df["state"].value_counts().sort_index()
        self._action_counts = self.df["action"].value_counts().sort_index()
        self._pair_counts = self.df.groupby(["state", "action"]).size()
        self._state_counts_df = self._counts_to_frame(self._state_counts, ["state"])
        self._action_counts_df = self._counts_to_frame(self._action_counts, ["action"])
        self._pair_counts_df = self._counts_to_frame(self._pair_counts, ["state", "action"])

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _counts_to_frame(counts, columns) -> pd.DataFrame:
        df = counts.reset_index(name="count")
        df.columns = [*columns, "count"]
        return df.astype({col: int for col in [*columns, "count"]})

    @staticmethod
    def _sort_counts(df: pd.DataFrame, columns, sort: bool) -> pd.DataFrame:
        if not sort:
            return df.reset_index(drop=True)
        sort_cols = ["count", *columns]
        ascending = [False, *([True] * len(columns))]
        return df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    @staticmethod
    def _is_pair(value) -> bool:
        return isinstance(value, tuple) and len(value) == 2

    @staticmethod
    def _normalize_action(action, action_map):
        if isinstance(action, str):
            if action_map is None:
                raise ValueError("action_map is required when querying string actions.")
            if action not in action_map:
                return None
            return int(action_map[action])
        return int(action)

    def _normalize_items(self, kind: str, items, action_map):
        if items is None:
            return None, False

        if kind == "pairs":
            if self._is_pair(items):
                action = self._normalize_action(items[1], action_map)
                return (int(items[0]), action), True
            pairs = []
            for item in items:
                if not self._is_pair(item):
                    raise ValueError("Pair queries must be (state, action) tuples.")
                action = self._normalize_action(item[1], action_map)
                pairs.append((int(item[0]), action))
            return pairs, False

        if isinstance(items, (str, bytes)):
            raise ValueError(f"{kind} queries must be integer labels, not strings.")

        if np.isscalar(items):
            return int(items), True

        return [int(item) for item in items], False

    @staticmethod
    def _require_dimension(value, name):
        if value is None:
            raise ValueError(f"{name} is required for include_missing=True.")
        return int(value)

    def _all_counts(self, kind: str, n_states, n_actions, include_missing: bool) -> pd.DataFrame:
        if kind == "states":
            df = self._state_counts_df.copy()
            if include_missing:
                n_states = self._require_dimension(n_states, "n_states")
                base = pd.DataFrame({"state": np.arange(n_states, dtype=int)})
                df = base.merge(df, on="state", how="left").fillna({"count": 0})
                df["count"] = df["count"].astype(int)
            return df

        if kind == "actions":
            df = self._action_counts_df.copy()
            if include_missing:
                n_actions = self._require_dimension(n_actions, "n_actions")
                base = pd.DataFrame({"action": np.arange(n_actions, dtype=int)})
                df = base.merge(df, on="action", how="left").fillna({"count": 0})
                df["count"] = df["count"].astype(int)
            return df

        if kind == "pairs":
            df = self._pair_counts_df.copy()
            if include_missing:
                n_states = self._require_dimension(n_states, "n_states")
                n_actions = self._require_dimension(n_actions, "n_actions")
                grid = pd.MultiIndex.from_product(
                    [range(n_states), range(n_actions)],
                    names=["state", "action"],
                ).to_frame(index=False)
                df = grid.merge(df, on=["state", "action"], how="left").fillna({"count": 0})
                df["count"] = df["count"].astype(int)
            return df

        raise ValueError("kind must be one of: 'states', 'actions', 'pairs'.")

    def counts(
        self,
        kind: str = "pairs",
        items=None,
        *,
        n_states: Optional[int] = None,
        n_actions: Optional[int] = None,
        include_missing: bool = False,
        action_map: Optional[dict] = None,
        sort: bool = True,
    ):
        """
        Count states, actions, or state-action pairs.

        ``items=None`` returns a DataFrame of all counts. A single item returns
        an integer. Multiple items return a DataFrame with a ``count`` column.
        """
        kind = kind.lower()
        columns = {"states": ["state"], "actions": ["action"], "pairs": ["state", "action"]}.get(kind)
        if columns is None:
            raise ValueError("kind must be one of: 'states', 'actions', 'pairs'.")

        normalized, single = self._normalize_items(kind, items, action_map)
        all_counts = self._all_counts(kind, n_states, n_actions, include_missing)

        if normalized is None:
            return self._sort_counts(all_counts, columns, sort)

        if single:
            if kind == "pairs":
                state, action = normalized
                if action is None:
                    return 0
                mask = (all_counts["state"] == state) & (all_counts["action"] == action)
            else:
                mask = all_counts[columns[0]] == normalized
            if not mask.any():
                return 0
            return int(all_counts.loc[mask, "count"].iloc[0])

        query = pd.DataFrame(normalized, columns=columns)
        query["count"] = [
            self.counts(kind, tuple(row) if kind == "pairs" else row[0], action_map=action_map)
            for row in query[columns].itertuples(index=False, name=None)
        ]
        return self._sort_counts(query, columns, sort)

    @staticmethod
    def _series_stats(counts: np.ndarray) -> dict:
        if counts.size == 0:
            return {
                "unique": 0,
                "min_count": 0,
                "max_count": 0,
                "mean_count": 0.0,
                "std_count": 0.0,
                "median_count": 0.0,
            }
        return {
            "unique": int(np.sum(counts > 0)),
            "min_count": int(counts.min()),
            "max_count": int(counts.max()),
            "mean_count": float(counts.mean()),
            "std_count": float(counts.std()),
            "median_count": float(np.median(counts)),
        }

    def stats(
        self,
        *,
        n_states: Optional[int] = None,
        n_actions: Optional[int] = None,
        include_missing: bool = True,
        top_n: int = 10,
        rare_n: int = 10,
    ) -> dict:
        """Return grouped dataset, frequency, coverage, and reward statistics."""
        pair_counts = self.counts(
            "pairs",
            n_states=n_states,
            n_actions=n_actions,
            include_missing=include_missing and n_states is not None and n_actions is not None,
        )
        state_counts = self.counts(
            "states",
            n_states=n_states,
            include_missing=include_missing and n_states is not None,
        )
        action_counts = self.counts(
            "actions",
            n_actions=n_actions,
            include_missing=include_missing and n_actions is not None,
        )

        result = {
            "dataset": {
                "source": self.source,
                "transitions": int(len(self.df)),
                "columns": list(self.df.columns),
            },
            "states": self._series_stats(state_counts["count"].to_numpy()),
            "actions": self._series_stats(action_counts["count"].to_numpy()),
            "pairs": self._series_stats(pair_counts["count"].to_numpy()),
            "coverage": None,
            "rewards": None,
            "top_pairs": pair_counts.head(top_n).reset_index(drop=True),
            "rare_pairs": pair_counts.sort_values(["count", "state", "action"]).head(rare_n).reset_index(drop=True),
            "missing_pairs": None,
        }

        if n_states is not None and n_actions is not None:
            total_possible = int(n_states) * int(n_actions)
            observed_pairs = int((pair_counts["count"] > 0).sum())
            result["coverage"] = {
                "total_possible_pairs": total_possible,
                "observed_pairs": observed_pairs,
                "missing_pairs": total_possible - observed_pairs,
                "coverage_percent": 100.0 * observed_pairs / total_possible if total_possible else 0.0,
            }
            if include_missing:
                result["missing_pairs"] = pair_counts.loc[
                    pair_counts["count"] == 0, ["state", "action"]
                ].reset_index(drop=True)

        if "reward" in self.df.columns:
            rewards = self.df["reward"].to_numpy(dtype=float)
            result["rewards"] = {
                "mean": float(rewards.mean()) if rewards.size else 0.0,
                "std": float(rewards.std()) if rewards.size else 0.0,
                "min": float(rewards.min()) if rewards.size else 0.0,
                "max": float(rewards.max()) if rewards.size else 0.0,
            }

        return result

    def compare(
        self,
        kind: str = "pairs",
        items=None,
        *,
        n_states: Optional[int] = None,
        n_actions: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compare state or pair counts against the global distribution."""
        kind = kind.lower()
        if kind not in {"states", "pairs"}:
            raise ValueError("compare supports only kind='states' or kind='pairs'.")

        baseline = self.counts(
            kind,
            n_states=n_states,
            n_actions=n_actions,
            include_missing=(n_states is not None if kind == "states" else n_states is not None and n_actions is not None),
        )

        if items is None:
            result = baseline.copy()
        else:
            include_missing = (
                n_states is not None
                if kind == "states"
                else n_states is not None and n_actions is not None
            )
            counted = self.counts(
                kind,
                items,
                n_states=n_states,
                n_actions=n_actions,
                include_missing=include_missing,
            )
            if isinstance(counted, int):
                columns = ["state"] if kind == "states" else ["state", "action"]
                values = [items] if kind == "states" else list(items)
                result = pd.DataFrame([values + [counted]], columns=[*columns, "count"])
            else:
                result = counted.copy()

        counts = baseline["count"].to_numpy(dtype=float)
        mean = float(counts.mean()) if counts.size else 0.0
        std = float(counts.std()) if counts.size else 0.0
        result["mean"] = mean
        result["std"] = std
        result["diff_from_mean"] = result["count"] - mean
        result["ratio_to_mean"] = result["count"] / mean if mean > 0 else np.inf
        result["z_score"] = (result["count"] - mean) / std if std > 0 else 0.0
        result["percentile"] = result["count"].apply(
            lambda count: 100.0 * np.sum(counts <= float(count)) / len(counts) if len(counts) else 0.0
        )
        return result.reset_index(drop=True)

    def summary(
        self,
        *,
        n_states: Optional[int] = None,
        n_actions: Optional[int] = None,
        print_result: bool = True,
    ) -> dict:
        """Return stats and optionally print a short readable summary."""
        result = self.stats(n_states=n_states, n_actions=n_actions)
        if print_result:
            dataset = result["dataset"]
            pairs = result["pairs"]
            print("=" * 50)
            print("Dataset Analysis Summary")
            print("=" * 50)
            print(f"Source: {dataset['source']}")
            print(f"Transitions: {dataset['transitions']:,}")
            print(f"Unique states: {result['states']['unique']:,}")
            print(f"Unique actions: {result['actions']['unique']:,}")
            print(f"Unique pairs: {pairs['unique']:,}")
            if result["coverage"] is not None:
                coverage = result["coverage"]
                print(f"Coverage: {coverage['coverage_percent']:.2f}%")
                print(f"Missing pairs: {coverage['missing_pairs']:,}")
            print(
                "Pair count min/mean/max: "
                f"{pairs['min_count']:,} / {pairs['mean_count']:.2f} / {pairs['max_count']:,}"
            )
            print("=" * 50)
        return result

    @staticmethod
    def _prepare_phi(phi, n_states, n_actions):
        phi_np = DatasetAnalyzer._to_numpy(phi).astype(float)
        if phi_np.ndim == 3:
            N, A, d = phi_np.shape
            return phi_np, phi_np.reshape(N * A, d), N, A, d
        if phi_np.ndim == 2:
            if n_states is None or n_actions is None:
                raise ValueError("n_states and n_actions are required when phi has shape (N*A, d).")
            N, A = int(n_states), int(n_actions)
            if phi_np.shape[0] != N * A:
                raise ValueError(f"phi has {phi_np.shape[0]} rows, expected n_states*n_actions={N * A}.")
            d = int(phi_np.shape[1])
            return phi_np.reshape(N, A, d), phi_np, N, A, d
        raise ValueError("phi must have shape (N*A, d) or (N, A, d).")

    @staticmethod
    def _prepare_occupancy(occupancy, expected_size):
        occ = DatasetAnalyzer._to_numpy(occupancy).reshape(-1).astype(float)
        if occ.shape != (expected_size,):
            raise ValueError(f"occupancy must have shape ({expected_size},), got {occ.shape}.")
        return occ

    def feature_coverage(
        self,
        phi,
        *,
        occupancy=None,
        optimal_occupancy=None,
        solver=None,
        beta: float = 0.0,
        n_states: Optional[int] = None,
        n_actions: Optional[int] = None,
        return_details: bool = False,
    ):
        """
        Compute ``lambda.T @ inv(Lambda_n) @ lambda`` for an explicit target occupancy.
        """
        if len(self.df) == 0:
            raise ValueError("Dataset is empty; cannot compute feature coverage.")

        if occupancy is not None:
            target_occupancy = occupancy
        elif optimal_occupancy is not None:
            target_occupancy = optimal_occupancy
        elif solver is not None and hasattr(solver, "mu_star"):
            target_occupancy = solver.mu_star
        else:
            raise ValueError("Provide occupancy, optimal_occupancy, or solver with mu_star.")

        phi_full, phi_flat, N, A, d = self._prepare_phi(phi, n_states, n_actions)
        occ = self._prepare_occupancy(target_occupancy, N * A)

        states = self.df["state"].to_numpy(dtype=int)
        actions = self.df["action"].to_numpy(dtype=int)
        bad_states = (states < 0) | (states >= N)
        bad_actions = (actions < 0) | (actions >= A)
        if np.any(bad_states) or np.any(bad_actions):
            raise ValueError("Dataset contains states/actions outside phi dimensions.")

        phi_data = phi_full[states, actions]
        n = int(len(self.df))
        covariance = float(beta) * np.eye(d) + (phi_data.T @ phi_data) / n
        lambda_target = phi_flat.T @ occ
        ratio = float(lambda_target.T @ np.linalg.solve(covariance, lambda_target))

        if return_details:
            return {
                "coverage_ratio": ratio,
                "covariance": covariance,
                "lambda": lambda_target,
                "occupancy": occ,
                "beta": float(beta),
                "n": n,
            }
        return ratio

    def __repr__(self) -> str:
        return (
            f"DatasetAnalyzer(source='{self.source}', "
            f"transitions={len(self.df)}, unique_pairs={len(self._pair_counts)})"
        )
