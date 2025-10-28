# Arquitetura da Aplicação

Este documento descreve a arquitetura do projeto `numerical-methods-for-pde`, detalhando os componentes, suas responsabilidades e as interações entre eles. O diagrama de arquitetura pode ser visualizado em [diagrams/application_architecture.md](diagrams/application_architecture.md).

## Visão Geral

A aplicação é estruturada de forma modular para separar as preocupações e facilitar a extensibilidade. A arquitetura pode ser dividida nas seguintes camadas principais:

1.  **Interface do Usuário (UI)**: Ponto de entrada para interação do usuário.
2.  **Aplicações**: Orquestradores que coordenam a execução dos solucionadores.
3.  **Solucionadores (Core Solvers)**: Lógica central para os solucionadores de sistemas lineares e não lineares.
4.  **Métodos Numéricos**: Implementações de algoritmos numéricos específicos.
5.  **Utilitários e Análise**: Módulos de suporte para tarefas como manipulação de dados, análise de matrizes e benchmarking.

---

## Componentes Detalhados

### 1. User Interface

#### `src/cli.py`
-   **Responsabilidade**: É o ponto de entrada principal da aplicação. Utiliza uma biblioteca de linha de comando (como `argparse` ou `click`) para interpretar os argumentos fornecidos pelo usuário.
-   **Interações**:
    -   Invoca as classes de aplicação (`LinearSolverApp` ou `NonlinearSolverApp`) com base nos comandos do usuário.
    -   Passa os parâmetros necessários, como o método a ser utilizado, o caminho do arquivo de entrada e outras configurações.

### 2. Applications

#### `src/app/linear_solver_app.py` e `src/app/nonlinear_solver_app.py`
-   **Responsabilidade**: Atuam como controladores que gerenciam o fluxo de execução para resolver um problema específico (linear ou não linear).
-   **Interações**:
    -   Recebem as solicitações da CLI.
    -   Utilizam os utilitários (`src/linear_solver/utils`) para carregar ou gerar as matrizes e vetores necessários.
    -   Instanciam o solucionador apropriado (`src/linear_solver` ou `src/nonlinear_solver`).
    -   Invocam o método de resolução e gerenciam os resultados.
    -   Podem utilizar o `MatrixAnalyzer` (`src/analysis/matrix_analyzer.py`) para verificar propriedades da matriz antes da execução.
    -   Usam utilitários gerais (`src/utils/files.py`) para salvar os resultados.

### 3. Core Solvers

#### `src/linear_solver/` e `src/nonlinear_solver/`
-   **Responsabilidade**: Contêm a lógica de alto nível para os solucionadores. As classes base (`base.py`) definem uma interface comum que todos os métodos de um determinado tipo devem seguir.
-   **Interações**:
    -   São instanciados pelas classes de `Application`.
    -   Selecionam e executam o método numérico específico (`src/linear_solver/methods` ou `src/nonlinear_solver/methods`) solicitado pelo usuário.
    -   Gerenciam o estado da solução, como o número de iterações e o erro calculado.

### 4. Numerical Methods

#### `src/linear_solver/methods/` e `src/nonlinear_solver/methods/`
-   **Responsabilidade**: Contêm as implementações concretas dos algoritmos numéricos. Cada arquivo corresponde a um método (ex: `jacobi.py`, `newton.py`).
-   **Interações**:
    -   São chamados pelos `Core Solvers`.
    -   Executam os cálculos matemáticos para encontrar a solução do sistema.
    -   Podem interagir com o `MatrixAnalyzer` para otimizar ou validar certas operações.

### 5. Utilities

#### `src/linear_solver/utils/`
-   **Responsabilidade**: Fornece ferramentas específicas para problemas lineares, como:
    -   `matrix_generator.py`: Gera matrizes com propriedades específicas (ex: diagonalmente dominante).
    -   `csv_loader.py`: Carrega matrizes e vetores de arquivos CSV.
    -   `matrix_validator.py`: Valida se uma matriz atende a certos critérios.

#### `src/utils/`
-   **Responsabilidade**: Contém utilitários genéricos que podem ser usados em todo o projeto, como `files.py` para operações de leitura e escrita de arquivos.

### 6. Analysis

#### `src/analysis/matrix_analyzer.py`
-   **Responsabilidade**: Oferece funções para analisar as propriedades de uma matriz, como verificar se é diagonalmente dominante, simétrica ou positiva definida.
-   **Interações**:
    -   É utilizado pelas `Applications` para guiar o usuário na escolha do método mais adequado.
    -   Pode ser usado por alguns `Numerical Methods` para garantir que as condições de convergência sejam atendidas.

### 7. Benchmarking

#### `src/benchmark/main.py`
-   **Responsabilidade**: Módulo dedicado a medir e comparar o desempenho (tempo de execução, número de iterações, etc.) dos diferentes métodos numéricos.
-   **Interações**:
    -   Invoca diretamente os `Core Solvers` e seus métodos com um conjunto de problemas predefinidos.
    -   Coleta e armazena os resultados para análise de desempenho.

---

## Fluxo de Execução (Exemplo Linear)

1.  O usuário executa o programa via `python src/cli.py linear --method jacobi --input data/matrix.csv`.
2.  O `cli.py` interpreta os argumentos e chama `LinearSolverApp`.
3.  `LinearSolverApp` usa `csv_loader.py` para carregar a matriz `A` e o vetor `b` do arquivo.
4.  `LinearSolverApp` pode usar `MatrixAnalyzer` para verificar se a matriz é adequada para o método de Jacobi.
5.  `LinearSolverApp` instancia o `LinearSolver`, que por sua vez seleciona o método `Jacobi`.
6.  O método `Jacobi` é executado, iterando até que a solução convirja.
7.  A solução é retornada para `LinearSolverApp`, que usa `files.py` para salvar o resultado em um arquivo.
