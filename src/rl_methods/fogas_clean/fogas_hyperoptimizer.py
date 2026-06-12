import inspect
import itertools
from pathlib import Path

import numpy as np


class FOGASHyperOptimizer:
    """
    Hyperparameter optimizer for FOGAS.

    The optimizer owns only the search loop. Metric construction is delegated to
    a FOGASEvaluator instance, so planner-dependent checks stay in the evaluator.
    """

    ALLOWED_PARAMETERS = {
        "alpha",
        "rho",
        "eta",
        "T",
        "D_theta",
        "c_min",
        "state_weight_update",
    }
    PARAMETER_ALIASES = {"et": "eta"}

    def __init__(self, solver, evaluator, metric, metric_kwargs=None, seed=42):
        self.solver = solver
        self.evaluator = evaluator
        self.metric_spec = metric
        self.metric_kwargs = dict(metric_kwargs or {})
        self.seed = seed

        if isinstance(metric, str):
            self.metric_name = metric
            self.metric = evaluator.get_metric(metric, **self.metric_kwargs)
        elif callable(metric):
            self.metric_name = getattr(metric, "__name__", metric.__class__.__name__)
            self.metric = metric
        else:
            raise TypeError("metric must be a string evaluator metric name or a callable.")

        if seed is not None:
            np.random.seed(seed)
            try:
                import torch
            except ImportError:
                torch = None
            if torch is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def optimize(
        self,
        mode="smart",
        parameters=("alpha", "rho", "eta"),
        values=None,
        search_space=None,
        smart_mode="sequential",
        strategy="bo",
        num_runs=1,
        coarse_points=10,
        bo_iters=15,
        random_candidates=30,
        fixed_params=None,
        plot=True,
        print_summary=True,
        top_k=5,
        progress=False,
        progress_leave=True,
        grid_n_jobs=1,
        results_output=None,
        **run_kwargs,
    ):
        mode = self._canonical_choice(mode, {"grid", "smart"}, "mode")
        strategy = self._canonical_choice(strategy, {"bo", "random"}, "strategy")
        smart_mode = self._canonical_choice(
            smart_mode, {"sequential", "joint"}, "smart_mode"
        )
        parameters = self._normalize_parameters(parameters)
        fixed_params = self._normalize_param_dict(fixed_params or {})
        search_space = self._normalize_search_space(search_space or {})
        run_kwargs = self._normalize_param_dict(run_kwargs)
        num_runs = int(num_runs)
        if num_runs < 1:
            raise ValueError("num_runs must be >= 1.")

        self._validate_tunable_params(parameters)
        self._validate_tunable_params(fixed_params.keys())
        self._validate_tunable_params(search_space.keys())

        history = []
        progress_bar = self._make_progress_bar(
            enabled=progress,
            mode=mode,
            smart_mode=smart_mode,
            strategy=strategy,
            parameters=parameters,
            values=values,
            coarse_points=int(coarse_points),
            bo_iters=int(bo_iters),
            random_candidates=int(random_candidates),
            leave=progress_leave,
        )
        try:
            if mode == "grid":
                result = self._optimize_grid(
                    parameters=parameters,
                    values=values,
                    fixed_params=fixed_params,
                    num_runs=num_runs,
                    history=history,
                    run_kwargs=run_kwargs,
                    progress_bar=progress_bar,
                )
                result["smart_mode"] = None
                result["strategy"] = None
            elif smart_mode == "sequential":
                result = self._optimize_smart_sequential(
                    parameters=parameters,
                    search_space=search_space,
                    strategy=strategy,
                    num_runs=num_runs,
                    coarse_points=int(coarse_points),
                    bo_iters=int(bo_iters),
                    random_candidates=int(random_candidates),
                    fixed_params=fixed_params,
                    history=history,
                    run_kwargs=run_kwargs,
                    progress_bar=progress_bar,
                )
                result["smart_mode"] = smart_mode
                result["strategy"] = strategy
            else:
                result = self._optimize_smart_joint(
                    parameters=parameters,
                    search_space=search_space,
                    strategy=strategy,
                    num_runs=num_runs,
                    coarse_points=int(coarse_points),
                    bo_iters=int(bo_iters),
                    random_candidates=int(random_candidates),
                    fixed_params=fixed_params,
                    history=history,
                    run_kwargs=run_kwargs,
                    progress_bar=progress_bar,
                )
                result["smart_mode"] = smart_mode
                result["strategy"] = strategy
        finally:
            if progress_bar is not None:
                progress_bar.close()

        result.update(
            {
                "mode": mode,
                "metric": self.metric_name,
                "history": history,
            }
        )
        result["history_df"] = self._history_frame(history)
        if results_output is not None:
            self._save_history_frame(result["history_df"], results_output)

        if print_summary:
            self._print_summary(result, parameters=parameters, top_k=top_k)
        if plot:
            self._plot_result(result, parameters=parameters)

        return result

    # ------------------------------------------------------------------
    # Search implementations
    # ------------------------------------------------------------------
    def _optimize_grid(
        self,
        parameters,
        values,
        fixed_params,
        num_runs,
        history,
        run_kwargs,
        progress_bar,
    ):
        if not values:
            raise ValueError("values is required when mode='grid'.")
        values = self._normalize_values(values)

        value_keys = list(values.keys())
        unknown = sorted(set(value_keys) - set(parameters))
        if unknown:
            raise ValueError(
                "Grid values were provided for parameter(s) not listed in "
                f"parameters: {unknown}."
            )

        base_params = self._base_params()
        base_params.update(fixed_params)
        base_params.update(run_kwargs)
        candidates = []

        for combo in itertools.product(*(values[p] for p in value_keys)):
            candidate = dict(base_params)
            candidate.update(dict(zip(value_keys, combo)))
            candidates.append(candidate)

        for candidate in candidates:
            self._evaluate_candidate(
                candidate,
                num_runs,
                history,
                stage="grid",
                progress_bar=progress_bar,
            )

        return self._best_result(history)

    def _optimize_smart_sequential(
        self,
        parameters,
        search_space,
        strategy,
        num_runs,
        coarse_points,
        bo_iters,
        random_candidates,
        fixed_params,
        history,
        run_kwargs,
        progress_bar,
    ):
        current = self._base_params()
        current.update(fixed_params)
        current.update(run_kwargs)
        self._evaluate_candidate(current, num_runs, history, stage="baseline", progress_bar=progress_bar)

        for parameter in parameters:
            cfg = self._space_for(parameter, search_space)
            grid = self._grid_from_space(cfg, coarse_points)
            stage_start = len(history)

            for value in grid:
                candidate = dict(current)
                candidate[parameter] = value
                self._evaluate_candidate(
                    candidate, num_runs, history, stage=f"coarse:{parameter}", progress_bar=progress_bar
                )

            stage_records = history[stage_start:]
            best_stage_record = min(stage_records, key=lambda item: item["metric"])
            best_value = best_stage_record["params"][parameter]
            left, right = self._neighbor_bounds(grid, best_value)

            if strategy == "random":
                candidates = self._random_values(cfg, random_candidates, left, right)
            else:
                candidates = self._bo_values(
                    cfg,
                    parameter=parameter,
                    left=left,
                    right=right,
                    records=stage_records,
                    bo_iters=bo_iters,
                    current=current,
                    num_runs=num_runs,
                    history=history,
                    progress_bar=progress_bar,
                )

            if strategy == "random":
                for value in candidates:
                    candidate = dict(current)
                    candidate[parameter] = value
                    self._evaluate_candidate(
                        candidate,
                        num_runs,
                        history,
                        stage=f"random:{parameter}",
                        progress_bar=progress_bar,
                    )

            stage_records = history[stage_start:]
            best_stage_record = min(stage_records, key=lambda item: item["metric"])
            current.update(self._strip_derived(best_stage_record["params"]))

        return self._best_result(history)

    def _optimize_smart_joint(
        self,
        parameters,
        search_space,
        strategy,
        num_runs,
        coarse_points,
        bo_iters,
        random_candidates,
        fixed_params,
        history,
        run_kwargs,
        progress_bar,
    ):
        base_params = self._base_params()
        base_params.update(fixed_params)
        base_params.update(run_kwargs)
        spaces = {p: self._space_for(p, search_space) for p in parameters}

        self._evaluate_candidate(base_params, num_runs, history, stage="baseline", progress_bar=progress_bar)

        if strategy == "random":
            for _ in range(random_candidates):
                candidate = dict(base_params)
                candidate.update(self._sample_joint(spaces))
                self._evaluate_candidate(
                    candidate, num_runs, history, stage="random:joint", progress_bar=progress_bar
                )
        else:
            for _ in range(coarse_points):
                candidate = dict(base_params)
                candidate.update(self._sample_joint(spaces))
                self._evaluate_candidate(candidate, num_runs, history, stage="init:joint", progress_bar=progress_bar)

            for _ in range(bo_iters):
                next_params = self._joint_bo_candidate(
                    parameters=parameters,
                    spaces=spaces,
                    records=history,
                    n_candidates=500,
                )
                candidate = dict(base_params)
                candidate.update(next_params)
                self._evaluate_candidate(candidate, num_runs, history, stage="bo:joint", progress_bar=progress_bar)

        return self._best_result(history)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _evaluate_candidate(self, params, num_runs, history, stage, progress_bar=None):
        params = self._complete_params(params)
        run_params = self._filter_solver_params(params)
        per_run = []

        for _ in range(num_runs):
            if hasattr(self.solver, "D_pi"):
                self.solver.D_pi = params["D_pi"]
            self.solver.run(**run_params)
            per_run.append(self._metric_value())

        metric = float(np.mean(per_run))
        record = {
            "stage": stage,
            "params": dict(params),
            "metric": metric,
            "per_run_metrics": per_run,
        }
        history.append(record)
        self._update_progress_bar(progress_bar, record, history)
        return record

    def _metric_value(self):
        value = self.metric()
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and isinstance(value, torch.Tensor):
            return float(value.item())
        return float(value)

    # ------------------------------------------------------------------
    # Progress reporting
    # ------------------------------------------------------------------
    def _make_progress_bar(
        self,
        enabled,
        mode,
        smart_mode,
        strategy,
        parameters,
        values,
        coarse_points,
        bo_iters,
        random_candidates,
        leave,
    ):
        if not enabled:
            return None

        try:
            from tqdm.auto import tqdm
        except ImportError:
            return None

        total = self._expected_evaluations(
            mode=mode,
            smart_mode=smart_mode,
            strategy=strategy,
            parameters=parameters,
            values=values,
            coarse_points=coarse_points,
            bo_iters=bo_iters,
            random_candidates=random_candidates,
        )
        return tqdm(total=total, desc="FOGAS hyperopt", unit="candidate", leave=leave)

    def _expected_evaluations(
        self,
        mode,
        smart_mode,
        strategy,
        parameters,
        values,
        coarse_points,
        bo_iters,
        random_candidates,
    ):
        if mode == "grid":
            if not values:
                return None
            normalized_values = self._normalize_values(values)
            total = 1
            for vals in normalized_values.values():
                total *= len(vals)
            return total

        if smart_mode == "sequential":
            refinement_points = random_candidates if strategy == "random" else bo_iters
            return 1 + len(parameters) * (coarse_points + refinement_points)

        if strategy == "random":
            return 1 + random_candidates
        return 1 + coarse_points + bo_iters

    def _update_progress_bar(self, progress_bar, record, history):
        if progress_bar is None:
            return

        best = min(history, key=lambda item: item["metric"])
        progress_bar.set_postfix(
            {
                "stage": record["stage"],
                "metric": f"{record['metric']:.3f}",
                "best": f"{best['metric']:.3f}",
            }
        )
        progress_bar.update(1)

    # ------------------------------------------------------------------
    # Bounds and candidate generation
    # ------------------------------------------------------------------
    def _default_search_space(self):
        base = self._base_params()
        t = int(base["T"])
        return {
            "alpha": {"bounds": (base["alpha"], 5.0), "log_scale": True},
            "rho": {"bounds": (1e-2, 5.0), "log_scale": True},
            "eta": {"bounds": (base["eta"], 3.0), "log_scale": True},
            "T": {
                "bounds": (max(1, int(0.25 * t)), max(1, int(2 * t))),
                "log_scale": False,
                "integer": True,
            },
            "D_theta": {
                "bounds": (0.25 * base["D_theta"], 4.0 * base["D_theta"]),
                "log_scale": True,
            },
        }

    def _space_for(self, parameter, search_space):
        cfg = dict(self._default_search_space()[parameter])
        override = search_space.get(parameter)
        if override is None:
            pass
        elif isinstance(override, (tuple, list)) and len(override) == 2:
            cfg["bounds"] = tuple(override)
        elif isinstance(override, dict):
            cfg.update(override)
        else:
            raise TypeError(
                f"search_space['{parameter}'] must be a (low, high) pair or a dict."
            )

        low, high = cfg["bounds"]
        low = float(low)
        high = float(high)
        if high < low:
            raise ValueError(f"Invalid bounds for {parameter}: high < low.")
        if cfg.get("log_scale", False) and low <= 0:
            raise ValueError(f"Log-scale bounds for {parameter} must be positive.")
        cfg["bounds"] = (low, high)
        cfg["integer"] = bool(cfg.get("integer", parameter == "T"))
        cfg["log_scale"] = bool(cfg.get("log_scale", False))
        return cfg

    def _grid_from_space(self, cfg, n_points):
        if n_points < 1:
            raise ValueError("coarse_points must be >= 1.")
        low, high = cfg["bounds"]
        if cfg["log_scale"]:
            values = np.logspace(np.log10(low), np.log10(high), n_points)
        else:
            values = np.linspace(low, high, n_points)
        return np.array([self._coerce_space_value(v, cfg) for v in values])

    def _random_values(self, cfg, n_candidates, left=None, right=None):
        if n_candidates < 0:
            raise ValueError("random_candidates must be >= 0.")
        low, high = cfg["bounds"] if left is None else (left, right)
        if cfg["log_scale"]:
            values = np.exp(np.random.uniform(np.log(low), np.log(high), n_candidates))
        else:
            values = np.random.uniform(low, high, n_candidates)
        return [self._coerce_space_value(v, cfg) for v in values]

    def _bo_values(
        self,
        cfg,
        parameter,
        left,
        right,
        records,
        bo_iters,
        current,
        num_runs,
        history,
        progress_bar=None,
    ):
        for _ in range(bo_iters):
            X = np.array(
                [[self._space_to_unit(record["params"][parameter], cfg)] for record in records]
            )
            y = np.array([record["metric"] for record in records])
            gp = self._fit_gp(X, y)

            candidate_values = self._candidate_line(cfg, left, right, 300)
            candidate_X = np.array(
                [[self._space_to_unit(value, cfg)] for value in candidate_values]
            )
            ei = self._expected_improvement(candidate_X, gp, np.min(y))
            value = candidate_values[int(np.argmax(ei))]

            candidate = dict(current)
            candidate[parameter] = value
            record = self._evaluate_candidate(
                candidate,
                num_runs,
                history,
                stage=f"bo:{parameter}",
                progress_bar=progress_bar,
            )
            records.append(record)

        return []

    def _joint_bo_candidate(self, parameters, spaces, records, n_candidates):
        X = np.array(
            [
                [self._space_to_unit(record["params"][p], spaces[p]) for p in parameters]
                for record in records
            ]
        )
        y = np.array([record["metric"] for record in records])
        gp = self._fit_gp(X, y)

        candidate_params = []
        candidate_X = []
        for _ in range(n_candidates):
            params = self._sample_joint(spaces)
            candidate_params.append(params)
            candidate_X.append(
                [self._space_to_unit(params[p], spaces[p]) for p in parameters]
            )
        candidate_X = np.array(candidate_X)
        ei = self._expected_improvement(candidate_X, gp, np.min(y))
        return candidate_params[int(np.argmax(ei))]

    @staticmethod
    def _fit_gp(X, y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

        kernel = (
            ConstantKernel(1.0)
            * Matern(length_scale=0.5, nu=2.5)
            + WhiteKernel(noise_level=0.05)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            optimizer=None,
        )
        gp.fit(X, y)
        return gp

    @staticmethod
    def _expected_improvement(X, gp, y_best, xi=0.01):
        from scipy.stats import norm

        mu, sigma = gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        z = (y_best - mu - xi) / sigma
        return (y_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)

    def _candidate_line(self, cfg, left, right, n_candidates):
        if cfg["log_scale"]:
            values = np.logspace(np.log10(left), np.log10(right), n_candidates)
        else:
            values = np.linspace(left, right, n_candidates)
        return [self._coerce_space_value(v, cfg) for v in values]

    def _sample_joint(self, spaces):
        return {
            parameter: self._random_values(cfg, 1)[0]
            for parameter, cfg in spaces.items()
        }

    def _neighbor_bounds(self, grid, best_value):
        idx = int(np.argmin(np.abs(np.asarray(grid, dtype=float) - float(best_value))))
        left = grid[max(idx - 1, 0)]
        right = grid[min(idx + 1, len(grid) - 1)]
        if left == right:
            return self._coerce_numeric(left), self._coerce_numeric(right)
        return self._coerce_numeric(left), self._coerce_numeric(right)

    def _space_to_unit(self, value, cfg):
        low, high = cfg["bounds"]
        value = self._coerce_numeric(value)
        if high == low:
            return 0.0
        if cfg["log_scale"]:
            return (np.log(value) - np.log(low)) / (np.log(high) - np.log(low))
        return (value - low) / (high - low)

    def _coerce_space_value(self, value, cfg):
        low, high = cfg["bounds"]
        value = min(max(float(value), low), high)
        if cfg["integer"]:
            value = int(round(value))
            value = int(min(max(value, int(round(low))), int(round(high))))
        return value

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------
    def _base_params(self):
        return {
            "alpha": float(self._solver_value("alpha")),
            "rho": float(self._solver_value("rho")),
            "eta": float(self._solver_value("eta")),
            "T": int(round(self._solver_value("T"))),
            "D_theta": float(self._solver_value("D_theta")),
        }

    def _solver_value(self, name):
        if name == "T":
            if hasattr(self.solver, "params") and hasattr(self.solver.params, "T"):
                return self.solver.params.T
            if hasattr(self.solver, "T"):
                return self.solver.T
        if hasattr(self.solver, name):
            return getattr(self.solver, name)
        if hasattr(self.solver, "params") and hasattr(self.solver.params, name):
            return getattr(self.solver.params, name)
        raise AttributeError(f"solver does not expose a default value for {name!r}.")

    def _complete_params(self, params):
        complete = self._base_params()
        complete.update(self._normalize_param_dict(params))
        complete["T"] = int(round(complete["T"]))
        complete["D_pi"] = (
            float(complete["alpha"]) * int(complete["T"]) * float(complete["D_theta"])
        )
        return complete

    @staticmethod
    def _strip_derived(params):
        return {k: v for k, v in params.items() if k != "D_pi"}

    def _normalize_parameters(self, parameters):
        if isinstance(parameters, str):
            parameters = [parameters]
        normalized = [self._normalize_parameter(p) for p in parameters]
        unknown = sorted(set(normalized) - self.ALLOWED_PARAMETERS)
        if unknown:
            raise ValueError(
                f"Unsupported hyperparameter(s): {unknown}. "
                f"Allowed: {sorted(self.ALLOWED_PARAMETERS)}."
            )
        return tuple(normalized)

    def _normalize_parameter(self, parameter):
        parameter = self.PARAMETER_ALIASES.get(parameter, parameter)
        return str(parameter)

    def _normalize_param_dict(self, params):
        return {
            self._normalize_parameter(key): value
            for key, value in dict(params).items()
        }

    def _normalize_search_space(self, search_space):
        return {
            self._normalize_parameter(key): value
            for key, value in dict(search_space).items()
        }

    def _normalize_values(self, values):
        normalized = {}
        for key, raw_values in self._normalize_param_dict(values).items():
            if key not in self.ALLOWED_PARAMETERS:
                raise ValueError(
                    f"Unsupported hyperparameter {key!r}. "
                    f"Allowed: {sorted(self.ALLOWED_PARAMETERS)}."
                )
            vals = list(raw_values)
            if not vals:
                raise ValueError(f"values['{key}'] must not be empty.")
            normalized[key] = vals
        return normalized

    def _solver_run_capabilities(self):
        signature = inspect.signature(self.solver.run)
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        accepted_params = {
            name
            for name, p in signature.parameters.items()
            if name != "self" and p.kind != inspect.Parameter.VAR_POSITIONAL
        }
        return accepts_var_kwargs, accepted_params

    def _filter_solver_params(self, params):
        return self._filter_solver_params_for_solver(self.solver, params)

    def _filter_solver_params_for_solver(self, solver, params):
        run_params = self._strip_derived(params)
        accepts_var_kwargs, accepted_params = self._solver_run_capabilities_for_solver(solver)
        if accepts_var_kwargs:
            return dict(run_params)
        return {key: value for key, value in run_params.items() if key in accepted_params}

    @staticmethod
    def _solver_run_capabilities_for_solver(solver):
        signature = inspect.signature(solver.run)
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        accepted_params = {
            name
            for name, p in signature.parameters.items()
            if name != "self" and p.kind != inspect.Parameter.VAR_POSITIONAL
        }
        return accepts_var_kwargs, accepted_params

    def _validate_tunable_params(self, param_names):
        param_names = tuple(param_names)
        unknown = sorted(set(param_names) - self.ALLOWED_PARAMETERS)
        if unknown:
            raise ValueError(
                f"Unsupported hyperparameter(s): {unknown}. "
                f"Allowed: {sorted(self.ALLOWED_PARAMETERS)}."
            )

        accepts_var_kwargs, accepted_params = self._solver_run_capabilities()
        if accepts_var_kwargs:
            return

        unsupported = sorted(key for key in param_names if key not in accepted_params)
        if unsupported:
            raise ValueError(
                "Solver does not accept runtime hyperparameter(s): "
                f"{unsupported}. Supported run parameters: {sorted(accepted_params)}"
            )

    @staticmethod
    def _canonical_choice(value, allowed, name):
        value = str(value).lower()
        if value not in allowed:
            raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}.")
        return value

    @staticmethod
    def _coerce_numeric(value):
        if isinstance(value, np.generic):
            return value.item()
        return value

    # ------------------------------------------------------------------
    # Results, summary, plots
    # ------------------------------------------------------------------
    @staticmethod
    def _best_result(history):
        if not history:
            raise ValueError("No candidates were evaluated.")
        best = min(history, key=lambda item: item["metric"])
        return {
            "best_params": dict(best["params"]),
            "best_metric": float(best["metric"]),
        }

    def _history_frame(self, history):
        import pandas as pd

        rows = []
        for idx, record in enumerate(history, start=1):
            row = {
                "run_idx": idx,
                "stage": record["stage"],
                "metric": float(record["metric"]),
            }
            row.update(record["params"])
            for run_idx, value in enumerate(record.get("per_run_metrics", []), start=1):
                row[f"per_run_metric_{run_idx}"] = float(value)
            rows.append(row)
        return pd.DataFrame(rows)

    def _save_history_frame(self, df, results_output):
        output_path = Path(results_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    @staticmethod
    def plot_saved_boxplot(
        csv_path,
        variable,
        metric_column="metric",
        variable_label=None,
        metric_label=None,
        title=None,
        sort_values=True,
        showmeans=True,
        figsize=(8, 4),
        ax=None,
    ):
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.read_csv(csv_path)
        if variable not in df.columns:
            raise ValueError(f"Column {variable!r} not found in {csv_path}.")
        if metric_column not in df.columns:
            raise ValueError(f"Column {metric_column!r} not found in {csv_path}.")

        grouped = df[[variable, metric_column]].dropna().groupby(variable)
        keys = list(grouped.groups.keys())
        if sort_values:
            keys = sorted(keys)

        data = [grouped.get_group(key)[metric_column].to_numpy() for key in keys]
        labels = [str(key) for key in keys]

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        ax.boxplot(data, labels=labels, showmeans=showmeans)
        ax.set_xlabel(variable_label or variable)
        ax.set_ylabel(metric_label or metric_column)
        ax.set_title(title or f"{metric_label or metric_column} by {variable_label or variable}")
        ax.grid(axis="y", alpha=0.3)
        return ax

    def _print_summary(self, result, parameters, top_k):
        history = result["history"]
        print("\n=== FOGAS Hyperparameter Optimization ===")
        print(f"Metric: {result['metric']}")
        print(f"Mode: {result['mode']}")
        if result.get("smart_mode") is not None:
            print(f"Smart mode: {result['smart_mode']} | Strategy: {result['strategy']}")
        print(f"Parameters: {tuple(parameters)}")
        print(f"Evaluated candidates: {len(history)}")
        print(f"Best metric: {result['best_metric']:.6g}")
        print(f"Best params: {self._format_params(result['best_params'])}")

        top = sorted(history, key=lambda item: item["metric"])[: int(top_k)]
        if top:
            print(f"\nTop {len(top)} candidates:")
            for rank, record in enumerate(top, start=1):
                print(
                    f"{rank}. metric={record['metric']:.6g} | "
                    f"stage={record['stage']} | "
                    f"{self._format_params(record['params'])}"
                )

    @staticmethod
    def _format_params(params):
        ordered = ["alpha", "rho", "eta", "T", "D_theta", "D_pi"]
        parts = []
        for key in ordered:
            if key in params:
                value = params[key]
                if key == "T":
                    parts.append(f"{key}={int(value)}")
                else:
                    parts.append(f"{key}={float(value):.4e}")
        return ", ".join(parts)

    def _plot_result(self, result, parameters):
        import matplotlib.pyplot as plt

        history = result["history"]
        if not history:
            return

        varied = [
            parameter
            for parameter in parameters
            if len({record["params"][parameter] for record in history}) > 1
        ]
        if not varied:
            return

        fig, axes = plt.subplots(
            len(varied),
            1,
            figsize=(8, max(4, 3 * len(varied))),
            squeeze=False,
        )

        for ax, parameter in zip(axes[:, 0], varied):
            grouped = {}
            for record in history:
                value = self._coerce_numeric(record["params"][parameter])
                grouped.setdefault(value, []).extend(
                    float(metric) for metric in record.get("per_run_metrics", [record["metric"]])
                )

            xs = sorted(grouped)
            means = []
            intervals = []
            for value in xs:
                samples = np.asarray(grouped[value], dtype=float)
                means.append(float(np.mean(samples)))
                if len(samples) > 1:
                    intervals.append(float(1.96 * np.std(samples, ddof=1) / np.sqrt(len(samples))))
                else:
                    intervals.append(0.0)

            ax.errorbar(xs, means, yerr=intervals, marker="o", capsize=4, linewidth=1.5)
            ax.set_xlabel(parameter)
            ax.set_ylabel("Metric")
            ax.set_title(f"{parameter} metric by tried value")
            ax.grid(True)

        plt.tight_layout()
        plt.show()
