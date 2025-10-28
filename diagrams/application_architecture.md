graph TD
    subgraph User Interface
        A[CLI - src/cli.py]
    end

    subgraph Applications
        B[Linear Solver App - src/app/linear_solver_app.py]
        C[Nonlinear Solver App - src/app/nonlinear_solver_app.py]
    end

    subgraph Core Solvers
        D[Linear Solver - src/linear_solver]
        E[Nonlinear Solver - src/nonlinear_solver]
    end

    subgraph Numerical Methods
        F[Linear Methods - src/linear_solver/methods]
        G[Nonlinear Methods - src/nonlinear_solver/methods]
    end

    subgraph Utilities
        H[Matrix Utilities - src/linear_solver/utils]
        I[General Utilities - src/utils]
        J[Matrix Analyzer - src/analysis/matrix_analyzer.py]
    end

    subgraph Benchmarking
        K[Benchmark - src/benchmark/main.py]
    end

    A --> B
    A --> C
    B --> D
    C --> E
    D --> F
    E --> G
    D --> H
    B --> J
    F --> J
    subgraph " "
    end
    B --> I
    C --> I
    K --> D
    K --> E
