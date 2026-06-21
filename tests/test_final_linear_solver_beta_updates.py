import pytest
import torch

from src.rl_methods.fogas_generalization_clean.solvers.final_linear_solver import (
    FinalLinearSolver,
)


def _solver(beta_update, beta_projection_radius=None):
    solver = FinalLinearSolver.__new__(FinalLinearSolver)
    solver.beta_update = FinalLinearSolver._canonical_beta_update(beta_update)
    solver.beta_projection_radius = beta_projection_radius
    solver.H = torch.tensor(
        [[2.0, 0.5], [0.5, 3.0]],
        dtype=torch.float64,
    )
    solver.H_inv = torch.linalg.inv(solver.H)
    solver._EPS = FinalLinearSolver._EPS
    return solver


def test_beta_update_canonicalizes_existing_and_legacy_names():
    assert FinalLinearSolver._canonical_beta_update("fogas_full") == "fogas_full"
    assert FinalLinearSolver._canonical_beta_update("fogas_diag") == "fogas_diag"
    assert FinalLinearSolver._canonical_beta_update("full") == "fogas_full"
    assert FinalLinearSolver._canonical_beta_update("diagonal") == "fogas_diag"
    assert FinalLinearSolver._canonical_beta_update("diag") == "fogas_diag"
    assert FinalLinearSolver._canonical_beta_update("gradient") == "projected_gradient"
    assert FinalLinearSolver._canonical_beta_update("best_response") == "fenchel_br"


def test_beta_update_rejects_unknown_name():
    with pytest.raises(ValueError, match="beta_update must be one of"):
        FinalLinearSolver._canonical_beta_update("not_an_update")


def test_fogas_full_matches_existing_shrunk_preconditioned_formula():
    solver = _solver("fogas_full")
    beta_t = torch.tensor([1.0, -2.0], dtype=torch.float64)
    beta_grad = torch.tensor([4.0, 8.0], dtype=torch.float64)
    eta = 0.1
    rho = 0.5

    beta_next, direction, diagnostics = solver._compute_beta_update(
        beta_t=beta_t,
        beta_grad=beta_grad,
        eta=eta,
        rho=rho,
    )

    expected_direction = solver.H_inv @ beta_grad
    expected_next = (beta_t + eta * expected_direction) / (1.0 + rho * eta)
    torch.testing.assert_close(direction, expected_direction)
    torch.testing.assert_close(beta_next, expected_next)
    assert diagnostics["beta_update"] == "fogas_full"


def test_fogas_diag_matches_existing_shrunk_diagonal_formula():
    solver = _solver("fogas_diag")
    beta_t = torch.tensor([1.0, -2.0], dtype=torch.float64)
    beta_grad = torch.tensor([4.0, 8.0], dtype=torch.float64)
    eta = 0.1
    rho = 0.5

    beta_next, direction, diagnostics = solver._compute_beta_update(
        beta_t=beta_t,
        beta_grad=beta_grad,
        eta=eta,
        rho=rho,
    )

    diag_h = torch.diagonal(solver.H)
    expected_direction = beta_grad / diag_h
    expected_next = (beta_t + eta * expected_direction) / (1.0 + rho * eta)
    torch.testing.assert_close(direction, expected_direction)
    torch.testing.assert_close(beta_next, expected_next)
    assert diagnostics["beta_diag_min"] == pytest.approx(float(diag_h.min()))
    assert diagnostics["beta_diag_max"] == pytest.approx(float(diag_h.max()))


def test_projected_gradient_without_radius_is_plain_gradient_ascent():
    solver = _solver("projected_gradient")
    beta_t = torch.tensor([1.0, -2.0], dtype=torch.float64)
    beta_grad = torch.tensor([4.0, 8.0], dtype=torch.float64)
    eta = 0.1

    beta_next, direction, diagnostics = solver._compute_beta_update(
        beta_t=beta_t,
        beta_grad=beta_grad,
        eta=eta,
        rho=10.0,
    )

    torch.testing.assert_close(direction, beta_grad)
    torch.testing.assert_close(beta_next, beta_t + eta * beta_grad)
    assert diagnostics["beta_projection_radius"] is None


def test_projected_gradient_applies_l2_projection_when_radius_is_set():
    radius = 1.0
    solver = _solver("projected_gradient", beta_projection_radius=radius)
    beta_t = torch.tensor([1.0, -2.0], dtype=torch.float64)
    beta_grad = torch.tensor([4.0, 8.0], dtype=torch.float64)
    eta = 0.1

    beta_next, _direction, diagnostics = solver._compute_beta_update(
        beta_t=beta_t,
        beta_grad=beta_grad,
        eta=eta,
        rho=10.0,
    )

    candidate = beta_t + eta * beta_grad
    expected_next = radius * candidate / torch.linalg.norm(candidate)
    torch.testing.assert_close(beta_next, expected_next)
    assert diagnostics["beta_projection_radius"] == radius


def test_fenchel_best_response_ignores_eta_rho_and_returns_preconditioned_signal():
    solver = _solver("fenchel_br")
    beta_t = torch.tensor([1.0, -2.0], dtype=torch.float64)
    beta_grad = torch.tensor([4.0, 8.0], dtype=torch.float64)

    beta_next, direction, _diagnostics = solver._compute_beta_update(
        beta_t=beta_t,
        beta_grad=beta_grad,
        eta=0.1,
        rho=10.0,
    )

    expected_next = solver.H_inv @ beta_grad
    torch.testing.assert_close(beta_next, expected_next)
    torch.testing.assert_close(direction, expected_next - beta_t)


def test_fenchel_mirror_interpolates_between_current_beta_and_best_response():
    solver = _solver("fenchel_mirror")
    beta_t = torch.tensor([1.0, -2.0], dtype=torch.float64)
    beta_grad = torch.tensor([4.0, 8.0], dtype=torch.float64)
    eta = 0.25

    beta_next, direction, _diagnostics = solver._compute_beta_update(
        beta_t=beta_t,
        beta_grad=beta_grad,
        eta=eta,
        rho=10.0,
    )

    best_response = solver.H_inv @ beta_grad
    expected_direction = best_response - beta_t
    expected_next = (1.0 - eta) * beta_t + eta * best_response
    torch.testing.assert_close(direction, expected_direction)
    torch.testing.assert_close(beta_next, expected_next)
