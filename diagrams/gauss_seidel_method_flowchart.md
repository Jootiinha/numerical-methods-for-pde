# Diagrama de Fluxo Aprimorado para o Método de Gauss-Seidel

Este diagrama detalha o Método de Gauss-Seidel. Diferente do Jacobi, este método utiliza os valores dos componentes da solução `x` recém-calculados na mesma iteração para acelerar a convergência. Isso cria uma dependência de dados que o torna inerentemente sequencial.

```mermaid
graph TD
    style Start fill:#f9f,stroke:#333,stroke-width:2px
    style End fill:#f9f,stroke:#333,stroke-width:2px
    style Check fill:#ccf,stroke:#333,stroke-width:2px

    Start[Início] --> A(Definir A, b, chute inicial x, tol, max_iter);
    
    subgraph "Loop Iterativo Principal"
        A --> Check{"Loop: k < max_iter"};
        Check -- "Verdadeiro" --> B(Salvar `x` atual em `x_anterior` para o teste de convergência);
        B --> C("Para cada componente `i` de 1 a n");
        C -- "Loop" --> D(Soma 1 = Σ(A[i,j] * x[j]) para j < i<br><small><i>Usa os valores de `x` recém-calculados nesta iteração</i></small>);
        D --> E(Soma 2 = Σ(A[i,j] * x_anterior[j]) para j > i<br><small><i>Usa os valores de `x` da iteração anterior</i></small>);
        E --> F(Atualizar o componente `i` da solução:<br>x[i] = (b[i] - Soma 1 - Soma 2) / A[i,i]);
        F --> C;
        C -- "Fim do Loop i" --> G{Verificar Convergência:<br>||x - x_anterior|| < tol?};
        G -- "Sim" --> H[Solução Encontrada: x];
        G -- "Não" --> I(Incrementar contador: k = k + 1);
        I --> Check;
    end

    Check -- "Falso (Atingiu max_iter)" --> J[Falha na Convergência];
    H --> End[Fim];
    J --> End;
