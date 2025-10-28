# Diagrama de Fluxo Aprimorado para a Eliminação de Gauss

Este diagrama descreve o processo da Eliminação de Gauss, um método direto para resolver `Ax = b`. O método primeiro transforma a matriz `A` em uma matriz triangular superior `U` (Fase de Eliminação) e depois resolve o sistema `Ux = c` por substituição retroativa (Fase de Substituição).

```mermaid
graph TD
    style Start fill:#f9f,stroke:#333,stroke-width:2px
    style End fill:#f9f,stroke:#333,stroke-width:2px
    style Phase fill:#e6f3ff,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5

    Start[Início] --> A(Construir a matriz aumentada [A|b]);

    subgraph "Fase 1: Eliminação Progressiva (Triangularização)"
        direction LR
        A --> B{"Para cada pivô k de 1 a n-1"};
        B --> C{"Para cada linha i abaixo do pivô (de k+1 a n)"};
        C --> D(Calcular o multiplicador:<br>m = A[i,k] / A[k,k]<br><small><i>Fator para zerar o elemento A[i,k]</i></small>);
        D --> E(Atualizar a linha i:<br>Linha[i] = Linha[i] - m * Linha[k]<br><small><i>Zera o elemento abaixo do pivô</i></small>);
        E --> C;
        C --> B;
    end

    B --> F(Resultado: Matriz triangular superior [U|c]);

    subgraph "Fase 2: Substituição Retroativa"
        direction LR
        F --> G(Calcular a última variável:<br>x[n] = c[n] / U[n,n]);
        G --> H{"Para cada linha i de n-1 até 1 (de baixo para cima)"};
        H --> I(Calcular a soma dos termos conhecidos:<br>S = Σ(U[i,j] * x[j]) para j > i);
        I --> J(Calcular a variável x[i]:<br>x[i] = (c[i] - S) / U[i,i]);
        J --> H;
    end
    
    H --> Solucao[Solução x encontrada];
    Solucao --> End[Fim];
