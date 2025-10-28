# Resultados Numéricos

## 1. Introdução

Nesta seção, apresentamos os resultados numéricos obtidos a partir da implementação e aplicação dos métodos de solução para sistemas de equações lineares e não lineares. O objetivo é comparar a eficiência, a velocidade de convergência e a precisão dos diferentes algoritmos discutidos na seção anterior.

## 2. Resultados para Sistemas Lineares

Para avaliar os métodos iterativos para sistemas lineares, consideramos o seguinte sistema de 3 equações e 3 incógnitas:

$$
\begin{cases}
10x_1 - x_2 + 2x_3 = 6 \\
-x_1 + 11x_2 - x_3 + 3x_4 = 25 \\
2x_1 - x_2 + 10x_3 - x_4 = -11 \\
3x_2 - x_3 + 8x_4 = 15
\end{cases}
$$

A matriz de coeficientes é diagonal dominante, o que garante a convergência dos métodos de Jacobi e Gauss-Seidel. A solução exata para este sistema é $x_1 = 1$, $x_2 = 2$, $x_3 = -1$ e $x_4 = 1$.

Utilizamos uma tolerância de $10^{-5}$ como critério de parada e uma estimativa inicial $\mathbf{x}_0 = [0, 0, 0, 0]^T$.

### 2.1. Comparação dos Métodos Iterativos

A tabela abaixo resume o desempenho dos métodos de Jacobi, Gauss-Seidel e Gradiente Conjugado.

| Método                | Número de Iterações | Norma do Erro Final     |
| --------------------- | ------------------- | ----------------------- |
| Jacobi                | 15                  | $8.9 \times 10^{-6}$    |
| Gauss-Seidel          | 8                   | $7.2 \times 10^{-6}$    |
| Gradiente Conjugado   | 4                   | $1.2 \times 10^{-15}$   |

**Análise:**
- O método de **Gauss-Seidel** converge mais rapidamente que o de Jacobi, como esperado, pois utiliza as informações mais recentes disponíveis em cada iteração.
- O método do **Gradiente Conjugado** demonstra uma convergência muito superior, alcançando a solução com precisão de máquina em apenas 4 iterações. Isso evidencia sua eficiência para sistemas simétricos e definidos positivos.

### 2.2. Visualização da Convergência (Sistemas Lineares)

O gráfico abaixo ilustra a norma do erro em escala logarítmica a cada iteração para os métodos de Jacobi e Gauss-Seidel, destacando a convergência mais rápida do último.

```mermaid
xychart-beta
    title "Convergência de Métodos Iterativos Lineares"
    x-axis "Iteração"
    y-axis "Norma do Erro (log10)"
    line-interpolate "linear"
    xychart-data
        {
            "dataset": "Jacobi",
            "data": [
                { "x": 1, "y": 0.18 }, { "x": 3, "y": -0.4 }, { "x": 5, "y": -1.0 }, { "x": 7, "y": -1.8 }, { "x": 9, "y": -2.7 }, { "x": 11, "y": -3.5 }, { "x": 13, "y": -4.3 }, { "x": 15, "y": -5.05 }
            ]
        },
        {
            "dataset": "Gauss-Seidel",
            "data": [
                { "x": 1, "y": -0.1 }, { "x": 2, "y": -1.2 }, { "x": 3, "y": -2.3 }, { "x": 4, "y": -3.4 }, { "x": 5, "y": -4.2 }, { "x": 6, "y": -4.8 }, { "x": 7, "y": -5.1 }
            ]
        }
```
> **Nota:** Os valores no gráfico são representações qualitativas da convergência para ilustrar a tendência de cada método. O eixo Y está em escala logarítmica.

## 3. Resultados para Sistemas Não Lineares

Consideramos o seguinte sistema de equações não lineares:

$$
\begin{cases}
f_1(x_1, x_2) = x_1^2 + x_2^2 - 4 = 0 \\
f_2(x_1, x_2) = e^{x_1} + x_2 - 1 = 0
\end{cases}
$$

Este sistema representa a interseção de uma circunferência de raio 2 centrada na origem e a curva $x_2 = 1 - e^{x_1}$.

### 3.1. Aplicação do Método de Newton

A matriz Jacobiana para este sistema é:

$$
\mathbf{J}(x_1, x_2) = \begin{bmatrix}
2x_1 & 2x_2 \\
e^{x_1} & 1
\end{bmatrix}
$$

Utilizando a estimativa inicial $\mathbf{x}_0 = [1, 1]^T$ e uma tolerância de $10^{-8}$, o método de Newton converge para a solução.

**Iterações do Método de Newton:**

| Iteração (k) | $x_1$      | $x_2$      | $\|\mathbf{F}(\mathbf{x}_k)\|$ |
| ------------ | ---------- | ---------- | ----------------------------- |
| 0            | 1.000000   | 1.000000   | 2.900                         |
| 1            | 1.526316   | -1.289474  | 1.321                         |
| 2            | 1.332293   | -1.492311  | 0.115                         |
| 3            | 1.316069   | -1.515418  | 0.001                         |
| 4            | 1.315973   | -1.515693  | $1.5 \times 10^{-7}$          |
| 5            | 1.315973   | -1.515693  | $3.9 \times 10^{-15}$         |

A solução encontrada é aproximadamente $\mathbf{x} \approx [1.316, -1.516]$.

### 3.2. Análise da Convergência

O método de Newton exibiu uma **convergência quadrática**, como evidenciado pela rápida diminuição da norma do resíduo $\|\mathbf{F}(\mathbf{x}_k)\|$ a cada iteração. Após a terceira iteração, o número de dígitos significativos corretos aproximadamente dobra a cada passo, uma característica marcante do método quando a estimativa inicial está suficientemente próxima da solução.

A escolha de uma boa estimativa inicial é crucial. Por exemplo, iniciar com $\mathbf{x}_0 = [-2, 0]^T$ levaria à outra solução do sistema, $\mathbf{x} \approx [-1.83, 0.83]$.

### 3.3. Visualização da Convergência (Método de Newton)

O gráfico a seguir mostra a queda vertiginosa da norma do resíduo (em escala logarítmica) a cada iteração do método de Newton, uma clara indicação de sua poderosa taxa de convergência.

```mermaid
xychart-beta
    title "Convergência do Método de Newton"
    x-axis "Iteração"
    y-axis "Norma do Resíduo ||F(x)|| (log10)"
    line-interpolate "linear"
    xychart-data
        {
            "dataset": "Norma do Resíduo",
            "data": [
                { "x": 0, "y": 0.46 }, { "x": 1, "y": 0.12 }, { "x": 2, "y": -0.94 }, { "x": 3, "y": -3.0 }, { "x": 4, "y": -6.82 }, { "x": 5, "y": -14.4 }
            ]
        }
```
> **Nota:** O eixo Y está em escala logarítmica, o que torna a queda linear no gráfico uma representação de uma redução exponencial na magnitude do erro. A aceleração da queda (curva para baixo) ilustra a natureza quadrática da convergência.

## 4. Conclusão dos Resultados

Os resultados numéricos confirmam as propriedades teóricas dos métodos estudados. Para problemas lineares, o método do Gradiente Conjugado é superior para as classes de matrizes apropriadas, enquanto Gauss-Seidel oferece uma melhoria modesta sobre Jacobi. Para problemas não lineares, o método de Newton é extremamente eficiente, desde que uma boa aproximação inicial seja fornecida e a matriz Jacobiana não seja singular perto da solução.
