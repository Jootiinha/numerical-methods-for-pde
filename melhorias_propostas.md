# Propostas de Melhoria para a Solução de Métodos Numéricos

Este documento descreve uma série de melhorias potenciais para o projeto, visando aumentar sua robustez, eficiência, usabilidade e capacidade de análise.

---

## 1. Aprimoramento dos Solvers Numéricos

### 1.1. Solvers Lineares
- **Pré-condicionadores Avançados:** O Método do Gradiente Conjugado (CG) é sensível ao mau condicionamento da matriz. A implementação de pré-condicionadores pode acelerar drasticamente a convergência.
    - **Sugestões:**
        - **Pré-condicionador de Jacobi:** Simples e eficaz para algumas matrizes.
        - **SSOR (Successive Over-Relaxation):** Geralmente mais eficaz que o de Jacobi.
        - **Incomplete Cholesky / Incomplete LU (IC/ILU):** Pré-condicionadores muito poderosos para uma vasta gama de problemas.
- **Suporte a Matrizes Esparsas:** Para problemas de grande escala, as matrizes são frequentemente esparsas.
    - **Sugestão:** Integrar o uso de formatos de matrizes esparsas (ex: `scipy.sparse.csr_matrix`) para reduzir o consumo de memória e o custo computacional das operações de matriz-vetor.

### 1.2. Solvers Não Lineares
- **Métodos Quasi-Newton:** O método de Newton requer o cálculo e a inversão (ou solução de um sistema linear) da matriz Jacobiana a cada passo, o que é caro.
    - **Sugestão:** Implementar métodos Quasi-Newton que aproximam a Jacobiana ou sua inversa, como:
        - **Método de Broyden:** Uma atualização de posto 1 para a Jacobiana.
        - **BFGS (Broyden–Fletcher–Goldfarb–Shanno):** Um dos métodos Quasi-Newton mais eficazes, que aproxima a inversa da Hessiana em problemas de otimização.
- **Estratégias de Globalização:** O método de Newton pode divergir se a estimativa inicial não for boa.
    - **Sugestão:** Implementar técnicas para garantir a convergência a partir de uma gama maior de estimativas iniciais:
        - **Busca Linear (Line Search):** Controla o tamanho do passo `α` na atualização `x_{k+1} = x_k + α * Δx` para garantir a diminuição do resíduo.
        - **Região de Confiança (Trust Region):** Resolve um subproblema de otimização em uma "região de confiança" onde o modelo linear ou quadrático é considerado válido.

## 2. Melhorias na Estrutura do Código e Usabilidade

- **Interface Unificada (Solver Factory):** Criar uma interface de alto nível que permita ao usuário selecionar o método de solução através de um parâmetro de string.
    - **Exemplo:** `solver = create_solver(method='cg', preconditioner='ilu')`
- **CLI (Command-Line Interface) Avançada:** Expandir a CLI (`src/cli.py`) para ser uma ferramenta de análise mais poderosa.
    - **Sugestões:**
        - Executar benchmarks comparativos: `python -m src.cli benchmark --methods jacobi,cg,pcg --problem-size 1000`
        - Gerar visualizações de convergência: `python -m src.cli plot-convergence --method newton --problem my_problem.json`
- **Arquivos de Configuração:** Externalizar parâmetros dos solvers, benchmarks e problemas para arquivos de configuração (ex: `config.yaml`). Isso evita hardcoding e facilita a experimentação.

## 3. Análise e Visualização de Resultados

- **Módulo de Benchmarking Sistemático:** Aprimorar o `src/benchmark/main.py` para executar baterias de testes, variando o tamanho do problema, o número de condição da matriz e outros parâmetros, salvando os resultados de forma estruturada (ex: CSV ou JSON).
- **Geração de Gráficos:** Criar scripts para gerar automaticamente visualizações a partir dos resultados dos benchmarks, como:
    - Gráficos de tempo de execução vs. tamanho da matriz.
    - Gráficos de número de iterações vs. número de condição.
    - Visualização da esparsidade das matrizes de teste.

## 4. Documentação e Testes

- **Documentação da API:** Utilizar o **Sphinx** para gerar uma documentação HTML a partir das docstrings do código. Isso cria um manual de referência profissional para o projeto.
- **Aumento da Cobertura de Testes:** Expandir os testes unitários para cobrir casos de borda, como:
    - Matrizes singulares ou mal condicionadas.
    - Cenários de não convergência.
    - Validação da precisão da solução contra solvers de referência (ex: `numpy.linalg.solve`).
