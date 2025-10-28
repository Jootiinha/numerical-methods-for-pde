# Diagrama de Fluxo Aprimorado para o Método do Gradiente Descendente

Este diagrama ilustra o Método do Gradiente Descendente, um algoritmo de otimização para encontrar o mínimo local de uma função `g(x)`. Para resolver um sistema não linear `F(x) = 0`, pode-se aplicá-lo à função objetivo `g(x) = 0.5 * ||F(x)||^2`, onde o mínimo de `g(x)` corresponde à solução do sistema.

```mermaid
graph TD
    style Start fill:#f9f,stroke:#333,stroke-width:2px
    style End fill:#f9f,stroke:#333,stroke-width:2px
    style Check fill:#ccf,stroke:#333,stroke-width:2px

    Start[Início] --> A(Definir função objetivo g(x), chute inicial x_k,<br>taxa de aprendizado α, tol, max_iter);
    
    subgraph "Loop de Otimização"
        A --> Check{"Loop: k < max_iter"};
        Check -- "Verdadeiro" --> B(Calcular o gradiente:<br>∇g(x_k)<br><small><i>Indica a direção de maior crescimento da função</i></small>);
        B --> C{Critério de Parada:<br>||∇g(x_k)|| < tol?};
        C -- "Sim (Gradiente próximo de zero)" --> D[Solução Encontrada: x_k];
        C -- "Não" --> E(Atualizar a solução:<br>x_{k+1} = x_k - α * ∇g(x_k)<br><small><i>Dá um passo na direção oposta ao gradiente</i></small>);
        E --> F(Atualizar para a próxima iteração:<br>x_k = x_{k+1});
        F --> G(Incrementar contador: k = k + 1);
        G --> Check;
    end

    Check -- "Falso (Atingiu max_iter)" --> H[Falha na Convergência];
    D --> End[Fim];
    H --> End;
