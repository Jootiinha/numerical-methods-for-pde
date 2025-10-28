# Análise de Desempenho de Métodos Numéricos

Este documento compara o desempenho de vários métodos numéricos para resolver sistemas de equações lineares e não lineares, com base nos benchmarks executados no projeto.

## 1. Métodos para Sistemas Lineares

A análise foi realizada no "Sistema Brasileiro 36x36" com uma tolerância de `1e-4`.

### 1.1. Comparação de Desempenho

| Método           | Tempo Médio (s) | Iterações Médias | Erro Final Médio | Taxa de Sucesso |
| ---------------- | --------------- | ---------------- | ---------------- | --------------- |
| Jacobi           | 0.0002          | 26.0             | 2.08e-04         | 100%            |
| Jacobi Ordem 2   | 0.0003          | 32.0             | 2.96e-04         | 100%            |
| Gauss-Seidel     | 0.0009          | 15.0             | 9.27e-05         | 100%            |
| SOR (ω=1.25)     | 0.0013          | 12.0             | 1.06e-05         | 100%            |

### 1.2. Conclusões

- **Velocidade vs. Iterações:** O método de **Jacobi** foi o mais rápido em tempo de execução médio, apesar de exigir mais iterações que Gauss-Seidel e SOR. Isso sugere que o custo por iteração de Jacobi é significativamente menor.
- **Eficiência:** O método **SOR (Successive Over-Relaxation)** com ω=1.25 foi o mais eficiente em termos de número de iterações, convergindo com apenas 12 iterações em média. Ele também alcançou o menor erro final, indicando a maior precisão entre os métodos testados.
- **Custo-Benefício:** Para este sistema específico, **Gauss-Seidel** e **SOR** oferecem um melhor equilíbrio entre velocidade e número de iterações. Embora Jacobi seja marginalmente mais rápido, ele requer quase o dobro de iterações, o que pode ser um fator em matrizes maiores.

### 1.3. Estimativa de Tempo de Máquina

A tabela a seguir apresenta o tempo de máquina estimado para obter resultados aceitáveis com cada método.

| Método           | Tempo Típico (s) | Classificação |
| ---------------- | ---------------- | ------------- |
| Jacobi           | 0.0002           | MUITO RÁPIDO  |
| Jacobi Ordem 2   | 0.0003           | MUITO RÁPIDO  |
| Gauss-Seidel     | 0.0009           | MUITO RÁPIDO  |
| SOR (ω=1.25)     | 0.0013           | MUITO RÁPIDO  |

Todos os métodos lineares testados são extremamente rápidos para este sistema de dimensão 36x36, com tempos de execução na ordem de microssegundos a um milissegundo.

## 2. Métodos para Sistemas Não Lineares

A análise foi realizada em um sistema de 3 equações não lineares com uma tolerância de `1e-4`. Foram testadas 5 aproximações iniciais diferentes para cada método.

### 2.1. Comparação de Desempenho

| Método    | Taxa de Convergência | Tempo Médio (s) | Iterações Médias (converg.) |
| --------- | -------------------- | --------------- | --------------------------- |
| Newton    | 100% (5/5)           | 0.0001          | 6.2                         |
| Iteração  | 0% (0/5)             | N/A             | N/A                         |
| Gradiente | 0% (0/5)             | N/A             | N/A                         |

*O tempo médio para o método de Newton foi calculado a partir das 5 execuções que convergiram. Os outros métodos não convergiram em nenhuma das tentativas.*

### 2.2. Conclusões

- **Robustez e Eficiência:** O **método de Newton** demonstrou ser extremamente robusto e eficiente, convergindo para uma solução em 100% das tentativas, a partir de diferentes pontos iniciais.
- **Falha de Convergência:** Tanto o **método de Iteração (Ponto Fixo)** quanto o **método do Gradiente** falharam em convergir para a solução dentro da tolerância especificada. Isso indica que, para este sistema específico e com os parâmetros utilizados, eles não são métodos viáveis. A falha pode ser devida à escolha do ponto inicial, à natureza do sistema ou aos parâmetros dos próprios métodos (como o fator de relaxação `alpha` ou o tamanho do passo).

### 2.3. Estimativa de Tempo de Máquina

| Método | Tempo Típico (s) | Classificação |
| ------ | ---------------- | ------------- |
| Newton | 0.0001           | MUITO RÁPIDO  |

O método de Newton é a única escolha confiável para este sistema, fornecendo resultados precisos em uma fração de milissegundo.

## 3. Conclusão Geral

A análise de desempenho revelou diferenças significativas entre os métodos:

- **Para sistemas lineares**, todos os métodos testados (Jacobi, Gauss-Seidel, SOR) foram eficazes e rápidos. A escolha entre eles pode depender do compromisso desejado entre velocidade de execução e número de iterações. O método **SOR** se destacou pela precisão e baixo número de iterações.
- **Para o sistema não linear**, o **método de Newton** foi inquestionavelmente superior, sendo o único a convergir de forma consistente e rápida. Os métodos de Iteração e Gradiente se mostraram inadequados para este problema específico.

Esses resultados sublinham a importância de escolher o método numérico correto com base nas características do problema a ser resolvido.
