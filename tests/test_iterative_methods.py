"""
Testes para os métodos iterativos (Jacobi e Gauss-Seidel).
"""

import numpy as np
import pytest

from src.linear_solver.methods import GaussSeidelSolver, JacobiSolver


class TestJacobiSolver:
    """Testes para o método de Jacobi."""

    def setup_method(self):
        """Configuração para cada teste."""
        self.solver = JacobiSolver(tolerance=1e-8, max_iterations=1000)

    def test_simple_system(self):
        """Teste com sistema simples bem condicionado."""
        A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]], dtype=float)
        b = np.array([3, 2, 3], dtype=float)

        x, info = self.solver.solve(A, b)

        assert info["converged"], "Método deveria convergir"
        assert info["iterations"] < 100, "Deveria convergir rapidamente"

        # Verificar qualidade da solução
        residual = np.linalg.norm(A @ x - b)
        assert residual < 1e-6, f"Resíduo muito alto: {residual}"

    def test_diagonal_dominant_matrix(self):
        """Teste com matriz diagonalmente dominante."""
        A = np.array([[10, 1, 1], [1, 10, 1], [1, 1, 10]], dtype=float)
        b = np.array([12, 12, 12], dtype=float)

        x, info = self.solver.solve(A, b)

        assert info["converged"], "Deveria convergir com matriz diag. dominante"

        # Verificar se está próximo da solução esperada [1, 1, 1]
        x_expected = np.array([1, 1, 1])
        error = np.linalg.norm(x - x_expected)
        assert error < 1e-6, f"Solução incorreta: {x}"

    def test_non_square_matrix(self):
        """Teste com matriz não quadrada (deve falhar)."""
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        b = np.array([1, 2], dtype=float)

        with pytest.raises(ValueError, match="quadrada"):
            self.solver.solve(A, b)

    def test_incompatible_dimensions(self):
        """Teste com dimensões incompatíveis."""
        A = np.array([[1, 2], [3, 4]], dtype=float)
        b = np.array([1, 2, 3], dtype=float)

        with pytest.raises(ValueError, match="incompatíveis"):
            self.solver.solve(A, b)

    def test_zero_diagonal(self):
        """Teste com elemento nulo na diagonal."""
        A = np.array([[0, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=float)
        b = np.array([1, 2, 3], dtype=float)

        with pytest.raises(ValueError, match="diagonal principal não-nula"):
            self.solver.solve(A, b)

    def test_custom_initial_guess(self):
        """Teste com aproximação inicial personalizada."""
        A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]], dtype=float)
        b = np.array([3, 2, 3], dtype=float)
        x0 = np.array([1, 1, 1], dtype=float)

        x, info = self.solver.solve(A, b, x0)

        assert info["converged"], "Deveria convergir com x0 customizado"
        assert len(info["convergence_history"]) > 0, "Deveria ter histórico"

    def test_convergence_history(self):
        """Teste do histórico de convergência."""
        A = np.array([[3, 1], [1, 3]], dtype=float)
        b = np.array([4, 4], dtype=float)

        x, info = self.solver.solve(A, b)

        assert "convergence_history" in info
        history = info["convergence_history"]
        assert len(history) > 0

        # Verificar se erro diminui (pelo menos no final)
        assert history[-1] < history[0], "Erro deveria diminuir"

    def test_order2_functionality(self):
        """Teste da funcionalidade de ordem 2 do Jacobi."""
        # Usando pesos que devem convergir
        solver_ord2 = JacobiSolver(
            tolerance=1e-8,
            max_iterations=1000,
            omega1=0.8,
            omega2=0.2,
            omega3=0.0,
        )
        A = np.array([[4, -1], [-1, 4]], dtype=float)
        b = np.array([3, 3], dtype=float)

        x, info = solver_ord2.solve(A, b)

        assert info["converged"]
        assert "Ordem 2" in info["method"]
        assert "parameters" in info
        assert "omega1" in info["parameters"]


class TestGaussSeidelSolver:
    """Testes para o método de Gauss-Seidel."""

    def setup_method(self):
        """Configuração para cada teste."""
        self.solver = GaussSeidelSolver(tolerance=1e-8, max_iterations=1000)

    def test_simple_system(self):
        """Teste com sistema simples."""
        A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]], dtype=float)
        b = np.array([3, 2, 3], dtype=float)

        x, info = self.solver.solve(A, b)

        assert info["converged"], "Gauss-Seidel deveria convergir"

        # Verificar qualidade da solução
        residual = np.linalg.norm(A @ x - b)
        assert residual < 1e-6, f"Resíduo muito alto: {residual}"

    def test_faster_than_jacobi(self):
        """Teste se Gauss-Seidel converge mais rápido que Jacobi."""
        A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]], dtype=float)
        b = np.array([3, 2, 3], dtype=float)

        jacobi = JacobiSolver(tolerance=1e-8, max_iterations=1000)
        gauss_seidel = GaussSeidelSolver(tolerance=1e-8, max_iterations=1000)

        x_j, info_j = jacobi.solve(A, b)
        x_gs, info_gs = gauss_seidel.solve(A, b)

        assert info_j["converged"] and info_gs["converged"]

        # Gauss-Seidel geralmente converge mais rápido
        # (não sempre garantido, mas para esta matriz específica)
        assert info_gs["iterations"] <= info_j["iterations"]

    def test_method_name(self):
        """Teste do nome do método."""
        assert self.solver.get_method_name() == "Gauss-Seidel"

    def test_sor_functionality(self):
        """Teste da funcionalidade SOR."""
        sor_solver = GaussSeidelSolver(
            tolerance=1e-8, max_iterations=1000, relaxation_factor=1.2
        )
        A = np.array([[4, -1], [-1, 4]], dtype=float)
        b = np.array([3, 3], dtype=float)

        x, info = sor_solver.solve(A, b)

        assert info["converged"]
        assert "SOR" in info["method"]
        assert "parameters" in info
        assert "relaxation_factor" in info["parameters"]
        assert info["parameters"]["relaxation_factor"] == 1.2

    def test_order2_functionality(self):
        """Teste da funcionalidade de ordem 2 do Gauss-Seidel."""
        solver_ord2 = GaussSeidelSolver(
            tolerance=1e-8,
            max_iterations=1000,
            relaxation_factor=1.1,
            omega1=0.9,
            omega2=0.1,
        )
        A = np.array([[4, -1], [-1, 4]], dtype=float)
        b = np.array([3, 3], dtype=float)

        x, info = solver_ord2.solve(A, b)

        assert info["converged"]
        assert "Ordem 2" in info["method"]
        assert "parameters" in info
        assert "omega1" in info["parameters"]
        assert "relaxation_factor" in info["parameters"]


class TestIterativeMethodsCommon:
    """Testes comuns para métodos iterativos."""

    @pytest.fixture
    def solvers(self):
        """Fixture com diferentes solvers."""
        return [
            JacobiSolver(tolerance=1e-6, max_iterations=100),
            GaussSeidelSolver(tolerance=1e-6, max_iterations=100),
        ]

    def test_non_convergent_system(self, solvers):
        """Teste com sistema que não converge."""
        # Matriz não diagonalmente dominante
        A = np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1]], dtype=float)
        b = np.array([6, 7, 8], dtype=float)

        for solver in solvers:
            x, info = solver.solve(A, b)

            # Pode ou não convergir dependendo da matriz
            # Importante é que retorne informação correta
            assert "converged" in info
            assert "iterations" in info
            assert "final_error" in info
            assert "method" in info

    def test_tolerance_parameter(self):
        """Teste do parâmetro de tolerância."""
        A = np.array([[4, -1], [-1, 4]], dtype=float)
        b = np.array([3, 3], dtype=float)

        # Tolerância mais restritiva deve resultar em mais iterações
        solver_loose = JacobiSolver(tolerance=1e-3, max_iterations=1000)
        solver_tight = JacobiSolver(tolerance=1e-8, max_iterations=1000)

        x1, info1 = solver_loose.solve(A, b)
        x2, info2 = solver_tight.solve(A, b)

        assert info1["converged"] and info2["converged"]
        assert info2["iterations"] >= info1["iterations"]
        assert info2["final_error"] <= info1["final_error"]

    def test_max_iterations_parameter(self):
        """Teste do parâmetro de máximo de iterações."""
        A = np.array([[4, -1], [-1, 4]], dtype=float)
        b = np.array([3, 3], dtype=float)

        # Limitar iterações artificialmente
        solver = JacobiSolver(tolerance=1e-12, max_iterations=5)

        x, info = solver.solve(A, b)

        # Provavelmente não vai convergir com tolerância muito baixa e poucas iterações
        assert info["iterations"] <= 5
        assert "converged" in info
