classDiagram
    class NonLinearSolver {
        <<abstract>>
        +solve(system_func, jacobian_func, x0) np.ndarray
        +get_method_name() str
    }

    class NewtonMethod {
        +solve(system_func, jacobian_func, x0) np.ndarray
        +get_method_name() str
    }

    class IterationMethod {
        +solve(system_func, jacobian_func, x0) np.ndarray
        +get_method_name() str
    }

    class GradientMethod {
        +solve(system_func, jacobian_func, x0) np.ndarray
        +get_method_name() str
    }

    NonLinearSolver <|-- NewtonMethod
    NonLinearSolver <|-- IterationMethod
    NonLinearSolver <|-- GradientMethod
