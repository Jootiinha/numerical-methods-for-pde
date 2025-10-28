# Diagrama de Fluxo Aprimorado para o Método do Gradiente Conjugado

Este diagrama de fluxo detalha o algoritmo do Gradiente Conjugado, um método iterativo altamente eficiente para resolver sistemas lineares `Ax = b` onde `A` é uma matriz simétrica e positiva definida. O método minimiza iterativamente uma função quadrática cuja solução coincide com a do sistema linear.

```mermaid
graph TD
    style Start fill:#f9f,stroke:#333,stroke-width:2px
    style End fill:#f9f,stroke:#333,stroke-width:2px
    style Check fill:#ccf,stroke:#333,stroke-width:2px
    
    Start[Início] --> A(Definir Matriz A, Vetor b, Chute Inicial x₀);
    
    subgraph "Fase de Inicialização"
        A --> B(Calcular resíduo inicial:<br>r₀ = b - A * x₀);
        B --> C(Definir direção inicial de busca:<br>p₀ = r₀);
        C --> D(Inicializar contador de iteração: k = 0);
    end

    D --> Check{"Loop Principal:<br>k < max_iter E ||rₖ|| > tol"};

    subgraph "Corpo do Loop Iterativo"
        Check -- "Verdadeiro" --> E(Calcular o tamanho do passo (αₖ):<br>αₖ = (rₖᵀ * rₖ) / (pₖᵀ * A * pₖ)<br><small><i>Move na direção pₖ para minimizar o erro</i></small>);
        E --> F(Atualizar a solução:<br>xₖ₊₁ = xₖ + αₖ * pₖ<br><small><i>Aproxima-se da solução</i></small>);
        F --> G(Atualizar o resíduo:<br>rₖ₊₁ = rₖ - αₖ * A * pₖ<br><small><i>Calcula o novo erro sem recalcular Ax</i></small>);
        G --> H(Calcular fator de correção da direção (βₖ):<br>βₖ = (rₖ₊₁ᵀ * rₖ₊₁) / (rₖᵀ * rₖ)<br><small><i>Garante que a nova direção seja A-ortogonal</i></small>);
        H --> I(Atualizar a direção de busca:<br>pₖ₊₁ = rₖ₊₁ + βₖ * pₖ<br><small><i>Combina a nova direção de descida com a anterior</i></small>);
        I --> J(Incrementar contador: k = k + 1);
        J --> Check;
    end

    Check -- "Falso (Convergiu ou Atingiu Limite)" --> End[Fim: Retornar xₖ];
