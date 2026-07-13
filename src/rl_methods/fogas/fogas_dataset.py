"""
FOGASDataset
------------

Discrete dataset adapter for FOGAS-format offline transitions.

This module does not generate data. It validates a saved CSV and exposes the
columns as tensors used by FOGASSolver, discrete generalized FOGAS solvers, and
FQI comparison code. Expected CSV columns:
    state, action, reward, next_state
"""

import csv

import torch
from typing import Union, List, Tuple, Optional

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal envs
    pd = None


class FOGASDataset:
    def __init__(self, csv_path, verbose=False):
        self.csv_path = csv_path
        required = ["state", "action", "reward", "next_state"]

        if pd is not None:
            self.df = pd.read_csv(csv_path)
            for col in required:
                if col not in self.df.columns:
                    raise ValueError(f"Missing required column: {col}")

            states = self.df["state"].to_numpy(dtype='int64')
            actions = self.df["action"].to_numpy(dtype='int64')
            rewards = self.df["reward"].to_numpy(dtype='float64')
            next_states = self.df["next_state"].to_numpy(dtype='int64')
        else:
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                raise ValueError("Dataset CSV is empty")
            missing = [col for col in required if col not in rows[0]]
            if missing:
                raise ValueError(f"Missing required column(s): {missing}")
            self.df = rows
            states = [int(row["state"]) for row in rows]
            actions = [int(row["action"]) for row in rows]
            rewards = [float(row["reward"]) for row in rows]
            next_states = [int(row["next_state"]) for row in rows]

        # Ensure action column is numeric if it's supposed to be loaded into torch
        # If it's strings, we might need a mapping before converting to tensor
        self.X = torch.as_tensor(states, dtype=torch.int64)
        self.A = torch.as_tensor(actions, dtype=torch.int64)
        self.R = torch.as_tensor(rewards, dtype=torch.float64)
        self.X_next = torch.as_tensor(next_states, dtype=torch.int64)

        self.n = len(self.df)

        if verbose:
            print(f"Loaded dataset {csv_path} with {self.n} transitions.")
            self.print_stats()

    def count_state(self, state: int) -> int:
        """Count how many times a state appears in the dataset."""
        return (self.X == state).sum().item()

    def count_states(self, states: List[int]) -> dict:
        """Count occurrences of multiple states."""
        return {s: self.count_state(s) for s in states}

    def count_pair(self, state: int, action: Union[int, str], action_map: Optional[dict] = None) -> int:
        """
        Count occurrences of a specific (state, action) pair.
        
        Parameters
        ----------
        state : int
            State index
        action : int or str
            Action index or action name (requires action_map)
        action_map : dict, optional
            Mapping from action names to indices (e.g., {'Down': 1})
        """
        if isinstance(action, str):
            if action_map is None:
                # Default mapping for standard GridWorld if mapping not provided
                action_map = {"Up": 0, "Down": 1, "Left": 2, "Right": 3}
            
            action_id = action_map.get(action)
            if action_id is None:
                return 0
            action = action_id
            
        return ((self.X == state) & (self.A == action)).sum().item()

    def count_pairs(self, pairs: List[Tuple[int, Union[int, str]]], action_map: Optional[dict] = None) -> dict:
        """Count occurrences of multiple (state, action) pairs."""
        return {p: self.count_pair(p[0], p[1], action_map) for p in pairs}

    def analyze_dataset(self, states: List[int] = None, pairs: List[Tuple[int, Union[int, str]]] = None, action_map: Optional[dict] = None):
        """
        Print advanced analysis of state visits and action pair frequencies.
        """
        print("\n" + "=" * 40)
        print("     ADVANCED DATASET ANALYSIS")
        print("=" * 40)
        
        if states:
            print("\n[State Visits]")
            counts = self.count_states(states)
            for s, c in counts.items():
                print(f"  State {s:2d}: {c:6d} visits")
                
        if pairs:
            print("\n[State-Action Pair Counts]")
            counts = self.count_pairs(pairs, action_map)
            for (s, a), c in counts.items():
                print(f"  Pair ({s}, {a}): {c:6d} occurrences")
        
        print("\n" + "=" * 40 + "\n")

    def summary(self):
        return {
            "n": self.n,
            "unique_states": torch.unique(self.X),
            "unique_actions": torch.unique(self.A),
            "reward_mean": float(self.R.mean().item()),
        }

    def print_stats(self):
        """
        Pretty-print basic statistics of the offline RL dataset.
        """
        print("\n========== FOGAS DATASET SUMMARY ==========\n")

        print(f"Total transitions (n): {self.n}")
        unique_states_count = len(torch.unique(self.X))
        unique_actions_count = len(torch.unique(self.A))
        print(f"Unique states:         {unique_states_count}")
        print(f"Unique actions:        {unique_actions_count}")

        if unique_states_count <= 20:  # Only print full distribution for small state spaces
            print("\nState distribution:")
            unique_states, counts_states = torch.unique(self.X, return_counts=True)
            for s, c in zip(unique_states, counts_states):
                print(f"  State {s.item()}: {c.item()} samples")

        print("\nAction distribution:")
        unique_actions, counts_actions = torch.unique(self.A, return_counts=True)
        for a, c in zip(unique_actions, counts_actions):
            print(f"  Action {a.item()}: {c.item()} samples")

        print(f"\nReward statistics:")
        print(f"  Mean reward:    {self.R.mean().item():.4f}")
        print(f"  Std deviation:  {self.R.std().item():.4f}")
        print(f"  Min reward:     {self.R.min().item():.4f}")
        print(f"  Max reward:     {self.R.max().item():.4f}")

        print("\nNext-state distribution (Top 10):")
        unique_next, counts_next = torch.unique(self.X_next, return_counts=True)
        # Sort by counts descending
        sorted_indices = torch.argsort(counts_next, descending=True)
        for i in range(min(10, len(sorted_indices))):
            idx = sorted_indices[i]
            print(f"  Next state {unique_next[idx].item()}: {counts_next[idx].item()} transitions")

        print("\n===========================================\n")
