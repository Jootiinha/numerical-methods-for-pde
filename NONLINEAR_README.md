# Extensão para Sistemas Não Lineares

Esta extensão adiciona capacidade de resolver sistemas de equações não lineares ao projeto de métodos numéricos para EDPs.

## Sistema Implementado

O sistema não linear específico implementado é:

```
F₁: (x-1)² + (y-1)² + (z-1)² = 1
F₂: 2x² + (y-1)² = 4z
F₃: 3x² + 2z² = 4y
```

Transformado para a forma F(x) = 0:
```
F₁(x,y,z) = (x-1)² + (y-1)² + (z-1)² - 1 = 0
F₂(x,y,z) = 2x² + (y-1)² - 4z = 0
F₃(x,y,z) = 3x² + 2z² - 4y = 0
```

## Métodos Implementados

### 1. Método de Newton-Raphson
- **Classe**: `NewtonSolver`
- **Descrição**: Usa o jacobiano analítico para convergência quadrática
- **Fórmula**: x_{k+1} = x_k - J(x_k)^{-1} * F(x_k)
- **Características**: 
  - Convergência rápida (quando converge)
  - Requer jacobiano
  - Pode usar diferenças finitas se jacobiano não fornecido

### 2. Método da Iteração de Ponto Fixo
- **Classe**: `IterationSolver`
- **Descrição**: Transforma F(x) = 0 em x = G(x)
- **Fórmula**: x_{k+1} = x_k - α * F(x_k)
- **Características**:
  - Simples de implementar
  - Convergência depende do parâmetro α
  - Testa múltiplos valores de α automaticamente

### 3. Método do Gradiente
- **Classe**: `GradientSolver`
- **Descrição**: Minimiza f(x) = (1/2) * ||F(x)||²
- **Fórmula**: x_{k+1} = x_k - α_k * ∇f(x_k)
- **Características**:
  - Sempre converge para mínimo local
  - Usa busca linear adaptativa
  - Mais lento, mas robusto

## Como Usar

### Via Linha de Comando

```bash
# Resolver o sistema não linear com tolerância padrão (1e-6)
python main.py --nonlinear

# Com tolerância personalizada
python main.py --nonlinear --tolerance 1e-8

# Com número máximo de iterações personalizado
python main.py --nonlinear --max-iterations 2000

# Gerar mapa de bacias de atração (plano 2D)
python main.py --nonlinear --basin-map

# Gerar mapa com resolução personalizada
python main.py --nonlinear --basin-map --basin-resolution 200
```

## Análise de Bacias de Atração

Foi implementada uma ferramenta de visualização para gerar o **mapa de bacias de atração**. Este mapa ajuda a entender o comportamento de um método iterativo (atualmente, o método de Newton) em relação a diferentes pontos de partida.

- **O que é?**: É uma imagem onde cada pixel corresponde a um "chute" inicial. A cor do pixel indica para qual das raízes do sistema o método convergiu a partir daquele chute.
- **Utilidade**: Ajuda a visualizar a sensibilidade do método à aproximação inicial. Regiões de cores complexas ou fractais podem indicar um comportamento caótico.
- **Implementação**: A análise é feita em um plano 2D (atualmente fixo em `z=0`) onde uma grade de pontos `(x, y)` é varrida. Para cada ponto, o método de Newton é executado para determinar a qual raiz ele converge.

### Via Código Python

```python
from nonlinear_solver import NewtonSolver, IterationSolver, GradientSolver
from nonlinear_example import NonLinearSystemExample

# Criar instância do exemplo
example = NonLinearSystemExample()

# Executar todos os métodos
results = example.run_all_methods(
    tolerance=1e-6,
    max_iterations=1000
)

# Ou usar métodos individuais
def system_func(x):
    # Sua implementação de F(x)
    pass

def jacobian_func(x):
    # Sua implementação de J(x)
    pass

# Método de Newton
newton_solver = NewtonSolver(tolerance=1e-6)
solution, info = newton_solver.solve(system_func, jacobian_func, x0)

# Método da Iteração
iteration_solver = IterationSolver(tolerance=1e-6)
solution, info = iteration_solver.solve(system_func, None, x0, alpha=0.1)

# Método do Gradiente
gradient_solver = GradientSolver(tolerance=1e-6)
solution, info = gradient_solver.solve(system_func, jacobian_func, x0)
```

## Resultados Típicos

Para o sistema implementado, os métodos encontram duas soluções principais:

1. **Solução 1**: x ≈ 0.649, y ≈ 0.365, z ≈ 0.312
2. **Solução 2**: x ≈ 1.330, y ≈ 1.938, z ≈ 1.105

### Performance dos Métodos

| Método | Taxa de Convergência | Iterações Típicas | Precisão |
|--------|---------------------|-------------------|----------|
| Newton | 100% (5/5) | 5-11 | ~1e-12 |
| Iteração | 0% (0/5) | Diverge | N/A |
| Gradiente | 100% (5/5) | 180-600 | ~1e-5 |

**Observações**:
- O **Método de Newton** é o mais eficiente para este sistema
- O **Método da Iteração** diverge com os parâmetros padrão (pode precisar de α menor)
- O **Método do Gradiente** é robusto mas mais lento

## Arquivos de Saída

Os resultados são salvos em:
- `results/nonlinear/`: Diretório principal
- `nonlinear_results_tol_X_TIMESTAMP.txt`: Relatório detalhado.
- `nonlinear_comparison_TIMESTAMP.png`: Gráficos de comparação de performance dos métodos.
- `basin_map_TIMESTAMP.png`: Mapa de bacias de atração (se solicitado via `--basin-map`).

## Estrutura do Código

```
nonlinear_solver/
├── __init__.py              # Módulo principal
├── base.py                  # Classe base NonLinearSolver
└── methods/
    ├── __init__.py
    ├── newton.py            # Método de Newton
    ├── iteration.py         # Método da Iteração
    └── gradient.py          # Método do Gradiente

nonlinear_example.py         # Exemplo específico do sistema
```

## Critérios de Convergência

Todos os métodos usam dois critérios de parada:

1. **Convergência da função**: ||F(x)|| < tolerância
2. **Convergência do passo**: ||x_{k+1} - x_k|| < tolerância

O método para quando qualquer um dos critérios é satisfeito.

## Extensões Futuras

- Suporte para sistemas não lineares genéricos via arquivo
- Mais métodos (Quasi-Newton, Broyden, etc.)
- Melhor análise de convergência
- Visualizações 3D das soluções
- Análise de estabilidade das soluções
