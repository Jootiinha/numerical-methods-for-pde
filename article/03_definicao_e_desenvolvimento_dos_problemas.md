# DEFINIÇÃO E DESENVOLVIMENTO DOS PROBLEMAS

## 1. Fundamentação Teórica

### 1.1. Sistemas de Equações Lineares

Um sistema de equações lineares é um conjunto de equações da forma:

$$
\begin{cases}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\
\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m
\end{cases}
$$

Este sistema pode ser representado de forma compacta na notação matricial como:

$$
\mathbf{A}\mathbf{x} = \mathbf{b}
$$

Onde:
- $\mathbf{A} \in \mathbb{R}^{m \times n}$ é a matriz de coeficientes
- $\mathbf{x} \in \mathbb{R}^{n}$ é o vetor de incógnitas
- $\mathbf{b} \in \mathbb{R}^{m}$ é o vetor de termos independentes

### 1.2. Sistemas de Equações Não Lineares

Um sistema de equações não lineares é um conjunto de equações onde pelo menos uma delas contém termos não lineares. A forma geral é:

$$
\begin{cases}
f_1(x_1, x_2, \ldots, x_n) = 0 \\
f_2(x_1, x_2, \ldots, x_n) = 0 \\
\vdots \\
f_n(x_1, x_2, \ldots, x_n) = 0
\end{cases}
$$

Ou, em notação vetorial:

$$
\mathbf{F}(\mathbf{x}) = \mathbf{0}
$$

Onde $\mathbf{F}: \mathbb{R}^n \rightarrow \mathbb{R}^n$ é uma função vetorial não linear.

## 2. Métodos para Sistemas Lineares

### 2.1. Classificação dos Métodos

Os métodos para resolver sistemas lineares podem ser classificados em duas categorias principais:

#### Métodos Diretos
Métodos que encontram a solução exata (a menos de erros de arredondamento) em um número finito de operações aritméticas. Exemplos incluem:
- Eliminação de Gauss
- Decomposição LU
- Decomposição QR
- Fatoração de Cholesky

#### Métodos Iterativos
Métodos que geram uma sequência de aproximações $\{\mathbf{x}^{(k)}\}$ que converge para a solução exata. São especialmente úteis para sistemas de grande porte e esparsos.

### 2.2. Métodos Iterativos Clássicos

#### Método de Jacobi

O método de Jacobi resolve o sistema $\mathbf{A}\mathbf{x} = \mathbf{b}$ isolando cada variável $x_i$ da equação $i$:

$$
x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j=1, j \neq i}^{n} a_{ij} x_j^{(k)} \right)
$$

**Condições de Convergência:**
- A matriz $\mathbf{A}$ deve ser diagonalmente dominante por linhas
- Ou o raio espectral da matriz de iteração deve ser menor que 1

**Vantagens:**
- Simplicidade de implementação
- Paralelização natural
- Baixo uso de memória

**Desvantagens:**
- Convergência lenta
- Não utiliza informações atualizadas na mesma iteração

#### Método de Gauss-Seidel

O método de Gauss-Seidel é uma modificação do método de Jacobi que utiliza os valores recém-calculados na mesma iteração:

$$
x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j=1}^{i-1} a_{ij} x_j^{(k+1)} - \sum_{j=i+1}^{n} a_{ij} x_j^{(k)} \right)
$$

**Vantagens sobre Jacobi:**
- Convergência mais rápida
- Menor uso de memória (não precisa armazenar $\mathbf{x}^{(k)}$ e $\mathbf{x}^{(k+1)}$ separadamente)

**Desvantagens:**
- Não pode ser paralelizado facilmente
- Ordem de atualização das variáveis afeta a convergência

### 2.3. Métodos de Alta Ordem

#### Método do Gradiente Conjugado

O método do Gradiente Conjugado é um método iterativo para sistemas simétricos e definidos positivos. Ele minimiza a função quadrática:

$$
\phi(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T\mathbf{A}\mathbf{x} - \mathbf{b}^T\mathbf{x}
$$

**Algoritmo:**
1. Inicialização: $\mathbf{x}^{(0)} = \mathbf{0}$, $\mathbf{r}^{(0)} = \mathbf{b}$, $\mathbf{p}^{(0)} = \mathbf{r}^{(0)}$
2. Para $k = 0, 1, 2, \ldots$:
   - $\alpha_k = \frac{(\mathbf{r}^{(k)})^T\mathbf{r}^{(k)}}{(\mathbf{p}^{(k)})^T\mathbf{A}\mathbf{p}^{(k)}}$
   - $\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + \alpha_k\mathbf{p}^{(k)}$
   - $\mathbf{r}^{(k+1)} = \mathbf{r}^{(k)} - \alpha_k\mathbf{A}\mathbf{p}^{(k)}$
   - $\beta_k = \frac{(\mathbf{r}^{(k+1)})^T\mathbf{r}^{(k+1)}}{(\mathbf{r}^{(k)})^T\mathbf{r}^{(k)}}$
   - $\mathbf{p}^{(k+1)} = \mathbf{r}^{(k+1)} + \beta_k\mathbf{p}^{(k)}$

**Características:**
- Convergência garantida em no máximo $n$ iterações para sistemas $n \times n$
- Convergência muito rápida para sistemas bem condicionados
- Requer apenas produtos matriz-vetor

#### Método do Gradiente Conjugado Quadrado (CGS)

O CGS é uma extensão do método do Gradiente Conjugado para matrizes não simétricas. Ele evita a necessidade de calcular $\mathbf{A}^T$ explicitamente.

**Características:**
- Aplicável a matrizes não simétricas
- Pode apresentar instabilidade numérica
- Requer mais memória que o Gradiente Conjugado clássico

### 2.4. Análise de Convergência

#### Critério de Convergência

Um método iterativo converge se:

$$
\lim_{k \to \infty} \|\mathbf{x}^{(k)} - \mathbf{x}^*\| = 0
$$

Onde $\mathbf{x}^*$ é a solução exata.

#### Taxa de Convergência

A taxa de convergência é definida como:

$$
\rho = \lim_{k \to \infty} \frac{\|\mathbf{x}^{(k+1)} - \mathbf{x}^*\|}{\|\mathbf{x}^{(k)} - \mathbf{x}^*\|}
$$

- $\rho < 1$: Convergência linear
- $\rho = 0$: Convergência superlinear
- Convergência quadrática: $\|\mathbf{x}^{(k+1)} - \mathbf{x}^*\| \leq C\|\mathbf{x}^{(k)} - \mathbf{x}^*\|^2$

## 3. Métodos para Sistemas Não Lineares

### 3.1. Desafios dos Sistemas Não Lineares

A resolução de sistemas não lineares apresenta desafios adicionais:

1. **Múltiplas Soluções**: Podem existir várias soluções, uma única solução ou nenhuma solução
2. **Sensibilidade à Estimativa Inicial**: A convergência depende fortemente da escolha de $\mathbf{x}^{(0)}$
3. **Custo Computacional**: Cada iteração pode envolver a resolução de um sistema linear
4. **Convergência Local**: Muitos métodos convergem apenas se a estimativa inicial estiver próxima da solução

### 3.2. Método de Newton-Raphson

O método de Newton é o mais conhecido e eficiente para sistemas não lineares. Ele lineariza o sistema em torno da estimativa atual:

$$
\mathbf{F}(\mathbf{x}^{(k)}) + \mathbf{J}(\mathbf{x}^{(k)})(\mathbf{x}^{(k+1)} - \mathbf{x}^{(k)}) = \mathbf{0}
$$

Onde $\mathbf{J}(\mathbf{x})$ é a matriz Jacobiana:

$$
J_{ij}(\mathbf{x}) = \frac{\partial f_i}{\partial x_j}(\mathbf{x})
$$

**Fórmula de Iteração:**
$$
\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - [\mathbf{J}(\mathbf{x}^{(k)})]^{-1} \mathbf{F}(\mathbf{x}^{(k)})
$$

**Características:**
- Convergência quadrática quando próximo da solução
- Requer cálculo da matriz Jacobiana
- Pode divergir se a estimativa inicial for inadequada

### 3.3. Método da Iteração de Ponto Fixo

Este método reescreve o sistema $\mathbf{F}(\mathbf{x}) = \mathbf{0}$ na forma $\mathbf{x} = \mathbf{G}(\mathbf{x})$ e aplica a iteração:

$$
\mathbf{x}^{(k+1)} = \mathbf{G}(\mathbf{x}^{(k)})
$$

Uma forma comum é:
$$
\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - \alpha \mathbf{F}(\mathbf{x}^{(k)})
$$

Onde $\alpha$ é um parâmetro de relaxação.

**Condições de Convergência:**
- $\|\mathbf{G}'(\mathbf{x})\| < 1$ em uma vizinhança da solução
- A escolha de $\alpha$ é crucial para a convergência

### 3.4. Método do Gradiente

Este método minimiza a função objetivo:
$$
g(\mathbf{x}) = \frac{1}{2} \|\mathbf{F}(\mathbf{x})\|^2
$$

**Fórmula de Iteração:**
$$
\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - \alpha_k \nabla g(\mathbf{x}^{(k)})
$$

Onde:
$$
\nabla g(\mathbf{x}) = \mathbf{J}(\mathbf{x})^T \mathbf{F}(\mathbf{x})
$$

**Características:**
- Sempre converge para um mínimo local
- Convergência mais lenta que Newton
- Mais robusto quanto à estimativa inicial

## 4. Implementação Computacional

### 4.1. Arquitetura da Biblioteca

A biblioteca desenvolvida segue uma arquitetura modular com as seguintes componentes principais:

#### Interface de Usuário (`src/cli.py`)
- Ponto de entrada principal da aplicação
- Interpretação de comandos da linha de comando
- Configuração de parâmetros de execução

#### Aplicações (`src/app/`)
- `linear_solver_app.py`: Orquestração para solucionadores lineares
- `nonlinear_solver_app.py`: Orquestração para solucionadores não lineares

#### Solucionadores Core
- `src/linear_solver/`: Lógica de alto nível para sistemas lineares
- `src/nonlinear_solver/`: Lógica de alto nível para sistemas não lineares

#### Implementações dos Métodos
- `src/linear_solver/methods/`: Implementações dos métodos iterativos
- `src/nonlinear_solver/methods/`: Implementações dos métodos não lineares

#### Utilitários e Análise
- `src/utils/`: Ferramentas de manipulação de arquivos
- `src/analysis/`: Análise de propriedades de matrizes
- `src/benchmark/`: Ferramentas de benchmarking

### 4.2. Características da Implementação

#### Interface Unificada
Todos os métodos implementados seguem uma interface comum:

```python
class BaseSolver:
    def solve(self, A, b, x0=None):
        """
        Resolve o sistema Ax = b
        
        Returns:
            solution: Vetor solução
            info: Dicionário com informações de convergência
        """
```

#### Monitoramento de Convergência
Cada método registra:
- Histórico de erros por iteração
- Número de iterações executadas
- Status de convergência
- Tempo de execução
- Erro final alcançado

#### Validação Automática
A biblioteca inclui validação automática de:
- Propriedades da matriz (simetria, definida positiva)
- Condicionamento do sistema
- Compatibilidade entre matriz e método escolhido

### 4.3. Sistemas de Teste

#### Sistema Brasileiro 36×36
Sistema linear de grande porte baseado em dados reais do sistema elétrico brasileiro, representando um problema típico de análise de redes elétricas.

#### Sistema Não Linear Tridimensional
Sistema específico implementado:
```
F₁(x,y,z) = (x-1)² + (y-1)² + (z-1)² - 1 = 0
F₂(x,y,z) = 2x² + (y-1)² - 4z = 0
F₃(x,y,z) = 3x² + 2z² - 4y = 0
```

Este sistema representa a interseção de superfícies geométricas e possui múltiplas soluções.

## 5. Critérios de Parada e Controle de Erro

### 5.1. Critérios para Sistemas Lineares

#### Critério de Resíduo Relativo
$$
\frac{\|\mathbf{r}^{(k)}\|}{\|\mathbf{b}\|} < \epsilon
$$

Onde $\mathbf{r}^{(k)} = \mathbf{b} - \mathbf{A}\mathbf{x}^{(k)}$ é o resíduo.

#### Critério de Incremento Relativo
$$
\frac{\|\mathbf{x}^{(k+1)} - \mathbf{x}^{(k)}\|}{\|\mathbf{x}^{(k+1)}\|} < \epsilon
$$

### 5.2. Critérios para Sistemas Não Lineares

#### Critério de Função
$$
\|\mathbf{F}(\mathbf{x}^{(k)})\| < \epsilon
$$

#### Critério de Incremento
$$
\|\mathbf{x}^{(k+1)} - \mathbf{x}^{(k)}\| < \epsilon
```

A implementação utiliza ambos os critérios, parando quando qualquer um deles é satisfeito.

## 6. Análise de Complexidade Computacional

### 6.1. Complexidade dos Métodos Lineares

| Método | Operações por Iteração | Memória | Convergência |
|--------|----------------------|---------|--------------|
| Jacobi | $O(n^2)$ | $O(n)$ | Linear |
| Gauss-Seidel | $O(n^2)$ | $O(n)$ | Linear |
| Gradiente Conjugado | $O(n^2)$ | $O(n)$ | Superlinear |

### 6.2. Complexidade dos Métodos Não Lineares

| Método | Operações por Iteração | Memória | Convergência |
|--------|----------------------|---------|--------------|
| Newton | $O(n^3)$ | $O(n^2)$ | Quadrática |
| Iteração | $O(n^2)$ | $O(n)$ | Linear |
| Gradiente | $O(n^2)$ | $O(n)$ | Linear |

A complexidade do método de Newton é dominada pela resolução do sistema linear $\mathbf{J}(\mathbf{x}^{(k)}) \Delta\mathbf{x} = -\mathbf{F}(\mathbf{x}^{(k)})$ a cada iteração.

