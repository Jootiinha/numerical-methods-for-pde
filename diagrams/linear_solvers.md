classDiagram
    class LinearSolver {
        <<abstract>>
        +solve(A, b, x0) np.ndarray
        +get_method_name() str
    }

    class JacobiSolver {
        +solve(A, b, x0) np.ndarray
        +get_method_name() str
    }
    note for JacobiSolver "Também implementa Jacobi Ordem 2"

    class GaussSeidelSolver {
        +solve(A, b, x0) np.ndarray
        +get_method_name() str
    }
    note for GaussSeidelSolver "Também implementa SOR (Successive Over-Relaxation)"

    class ConjugateGradientSolver {
        +solve(A, b, x0) np.ndarray
        +get_method_name() str
    }

    class CGSSolver {
        +solve(A, b, x0) np.ndarray
        +get_method_name() str
    }

    class PreconditionedConjugateGradientSolver {
        +solve(A, b, x0) np.ndarray
        +get_method_name() str
    }

    LinearSolver <|-- JacobiSolver
    LinearSolver <|-- GaussSeidelSolver
    LinearSolver <|-- ConjugateGradientSolver
    LinearSolver <|-- CGSSolver
    LinearSolver <|-- PreconditionedConjugateGradientSolver
