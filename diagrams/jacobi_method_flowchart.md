# Diagrama de Fluxo Aprimorado para o Método de Jacobi

Este diagrama ilustra o Método de Jacobi, um algoritmo iterativo para resolver sistemas lineares `Ax = b`. A principal característica do método é que, para calcular a nova aproximação `x_novo`, ele utiliza exclusivamente os valores da aproximação anterior, `x_k`. Isso o torna altamente paralelizável.

```mermaid
graph TD
    style Start fill:#f9f,stroke:#333,stroke-width:2px
    style End fill:#f9f,stroke:#333,stroke-width:2px
    style Check fill:#ccf,stroke:#333,stroke-width:2px

    Start[Início] --> A(Definir A, b, chute inicial x_k, tol, max_iter);
    
    subgraph "Loop Iterativo Principal"
        A --> Check{"Loop: k < max_iter"};
        Check -- "Verdadeiro" --> B(Criar um novo vetor `x_novo` para a próxima aproximação);
        B --> C("Para cada componente `i` de 1 a n");
        C -- "Loop" --> D(Calcular a soma: S = Σ(A[i,j] * x_k[j]) para j ≠ i);
        D --> E(Calcular o novo valor para o componente `i`:<br>x_novo[i] = (b[i] - S) / A[i,i]<br><small><i>Usa apenas valores da iteração anterior `x_k`</i></small>);
        E --> C;
        C -- "Fim do Loop i" --> F{Verificar Convergência:<br>||x_novo - x_k|| < tol?};
        F -- "Sim" --> G[Solução Encontrada: x_novo];
        F -- "Não" --> H(Atualizar para a próxima iteração:<br>x_k = x_novo);
        H --> I(Incrementar contador: k = k + 1);
        I --> Check;
    end

    Check -- "Falso (Atingiu max_iter)" --> J[Falha na Convergência];
    G --> End[Fim];
    J --> End;
