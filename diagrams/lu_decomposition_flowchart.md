# Diagrama de Fluxo Aprimorado para a Decomposição LU

Este diagrama ilustra a solução de um sistema `Ax = b` através da Decomposição LU. O método é dividido em três etapas principais:
1.  **Fatoração:** A matriz `A` é decomposta em `L` (triangular inferior) e `U` (triangular superior).
2.  **Substituição Direta:** Resolve-se `Ly = b` para encontrar um vetor intermediário `y`.
3.  **Substituição Reversa:** Resolve-se `Ux = y` para encontrar a solução final `x`.

```mermaid
graph TD
    style Start fill:#f9f,stroke:#333,stroke-width:2px
    style End fill:#f9f,stroke:#333,stroke-width:2px
    style Error fill:#ffcccc,stroke:#cc0000
    
    Start[Início: Resolver Ax = b] --> Fatoracao;

    subgraph "Fase 1: Fatoração"
        Fatoracao(Decompor A em L e U, tal que A = LU);
        Fatoracao --> Check{Fatoração bem-sucedida?};
    end

    Check -- "Não (e.g., pivô zero)" --> Error[Erro: Decomposição falhou];
    Check -- "Sim" --> SubstDireta;

    subgraph "Fase 2: Substituição Direta (Forward Substitution)"
        SubstDireta(Resolver Ly = b para y<br><small><i>y[i] = (b[i] - Σ(L[i,j]*y[j])) / L[i,i] para j < i</i></small>);
    end

    SubstDireta --> SubstReversa;

    subgraph "Fase 3: Substituição Reversa (Backward Substitution)"
        SubstReversa(Resolver Ux = y para x<br><small><i>x[i] = (y[i] - Σ(U[i,j]*x[j])) / U[i,i] para j > i</i></small>);
    end

    SubstReversa --> Solucao[Solução Final x Encontrada];
    Solucao --> End[Fim];
    Error --> End;
