# Linear Solver - Métodos Numéricos para Sistemas Lineares

Uma biblioteca Python abrangente para resolução de sistemas lineares usando métodos numéricos iterativos, desenvolvida com foco em boas práticas de desenvolvimento de software.

## 🚀 Características

### Métodos Implementados

- **Métodos Iterativos Clássicos:**
  - Método de Jacobi
  - Método de Gauss-Seidel

- **Métodos de Alta Ordem:**
  - Método de Jacobi de Ordem 2 (com combinação de iterações)
  - Método de Gauss-Seidel de Ordem 2 (SOR de ordem 2)

- **Métodos do Gradiente:**
  - Gradiente Conjugado
  - Gradiente Conjugado Precondicionado

### Recursos

✅ **Carregamento de dados**: Suporte completo para arquivos CSV  
✅ **Análise de matrizes**: Validação automática de propriedades (simetria, definida positiva, etc.)  
✅ **Monitoramento de convergência**: Histórico completo do processo iterativo  
✅ **Geração de matrizes de teste**: Utilitários para criar diferentes tipos de sistemas  
✅ **Interface consistente**: API uniforme para todos os métodos  
✅ **Documentação completa**: Código bem documentado com exemplos  
✅ **Qualidade de código**: Pre-commit hooks, formatação automática, type hints  
✅ **Ambiente reprodutível**: Gerenciamento de dependências com Poetry  

## 📦 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- Poetry (gerenciador de dependências) - [Instruções de instalação](https://python-poetry.org/docs/#installation)

### Instalação com Poetry (Recomendado)

```bash
# Clone o repositório
git clone https://github.com/joaomonteiro/numerical-methods-for-pde.git
cd numerical-methods-for-pde

# Instale as dependências básicas
poetry install

# Para desenvolvimento (inclui ferramentas de teste e qualidade)
poetry install --with dev

# Para análise avançada (inclui Jupyter, Seaborn, etc.)
poetry install --with dev,analysis

# Ativar o ambiente virtual
poetry shell
```

### Instalação alternativa via pip

```bash
# Instalar diretamente do código fonte
pip install .

# Ou em modo desenvolvimento
pip install -e .
```

### Comandos Úteis do Makefile

```bash
# Instalar todas as dependências
make install-all

# Executar testes
make test

# Executar testes com cobertura
make test-cov

# Formatar código
make format

# Verificar qualidade do código
make lint

# Verificação completa (format + lint + test)
make check
```

## 🔧 Uso

A aplicação agora é executada através do `main.py`, que oferece uma interface de linha de comando (CLI) para acessar todas as funcionalidades.

### Executando a Aplicação (CLI)

Após a instalação, você pode executar a aplicação diretamente do terminal.

**Resolver todos os sistemas com todos os métodos aplicáveis:**
```bash
python main.py --all
```

**Resolver um sistema específico com métodos selecionados:**
```bash
# Usando Jacobi e Gauss-Seidel no sistema brasileiro
python main.py --jacobi --gauss-seidel --system "Sistema Brasileiro 36x36"
```

**Executar o modo de benchmark (análise de performance):**
```bash
python main.py --benchmark
```

**Resolver o sistema não linear:**
```bash
python main.py --nonlinear
```

**Ver todas as opções disponíveis:**
```bash
python main.py --help
```

### Uso como Biblioteca

O código também pode ser importado e utilizado como uma biblioteca em outros projetos Python. Os exemplos abaixo demonstram como usar os componentes individuais.

### 1. Carregando um sistema de arquivo CSV

```python
from linear_solver import CSVMatrixLoader

# Carregar matriz aumentada [A|b] de um arquivo CSV
A, b = CSVMatrixLoader.load_augmented_matrix("sistema.csv")

# Ou carregar de arquivos separados
A, b = CSVMatrixLoader.load_separate_files("matriz_A.csv", "vetor_b.csv")
```

### 2. Resolvendo com diferentes métodos

```python
from linear_solver import (
    JacobiSolver, GaussSeidelSolver, 
    ConjugateGradientSolver,
    JacobiOrder2Solver,
    PreconditionedConjugateGradientSolver
)

# Método de Jacobi
solver_jacobi = JacobiSolver(tolerance=1e-6, max_iterations=1000)
x_jacobi, info_jacobi = solver_jacobi.solve(A, b)

# Método de Gauss-Seidel
solver_gs = GaussSeidelSolver(tolerance=1e-6, max_iterations=1000)
x_gs, info_gs = solver_gs.solve(A, b)

# Gradiente Conjugado (para matrizes simétricas e positivas definidas)
solver_cg = ConjugateGradientSolver(tolerance=1e-8)
x_cg, info_cg = solver_cg.solve(A, b)

# Jacobi de Ordem 2 com parâmetros personalizados
solver_j2 = JacobiOrder2Solver(omega1=0.8, omega2=0.15, omega3=0.05)
x_j2, info_j2 = solver_j2.solve(A, b)

# Gradiente Conjugado Precondicionado
solver_pcg = PreconditionedConjugateGradientSolver(preconditioner="jacobi")
x_pcg, info_pcg = solver_pcg.solve(A, b)
```

### 3. Analisando resultados

```python
# Verificar convergência
if info_jacobi['converged']:
    print(f"Convergiu em {info_jacobi['iterations']} iterações")
    print(f"Erro final: {info_jacobi['final_error']:.2e}")
else:
    print("Não convergiu no limite de iterações")

# Acessar histórico de convergência
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.semilogy(info_jacobi['convergence_history'], label='Jacobi')
plt.semilogy(info_gs['convergence_history'], label='Gauss-Seidel')
plt.xlabel('Iteração')
plt.ylabel('Erro')
plt.legend()
plt.title('Convergência dos Métodos')
plt.grid(True)
plt.show()
```

## 📁 Formato dos Arquivos CSV

### Matriz Aumentada (exemplo: `sistema.csv`)
```csv
a11,a12,a13,b1
a21,a22,a23,b2  
a31,a32,a33,b3
```

### Arquivos Separados

**matriz_A.csv:**
```csv
a11,a12,a13
a21,a22,a23
a31,a32,a33
```

**vetor_b.csv:**
```csv
b1
b2
b3
```

## 🧪 Validação e Análise de Matrizes

```python
from linear_solver import MatrixValidator

# Analisar propriedades da matriz
analysis = MatrixValidator.analyze_matrix(A)

print(f"Matriz é simétrica: {analysis['is_symmetric']}")
print(f"Matriz é positiva definida: {analysis['is_positive_definite']}")
print(f"Matriz é diagonalmente dominante: {analysis['is_diagonally_dominant']}")
print(f"Número de condição: {analysis['condition_number']:.2e}")
```

## 🎯 Gerando Matrizes de Teste

```python
from linear_solver import MatrixGenerator

# Matriz diagonalmente dominante
A_dd, b_dd = MatrixGenerator.diagonally_dominant_matrix(
    n=5, dominance_factor=3.0, random_seed=42
)

# Matriz simétrica e positiva definida  
A_spd, b_spd = MatrixGenerator.symmetric_positive_definite_matrix(
    n=4, condition_number=100.0, random_seed=42
)

# Matriz tridiagonal
A_tri, b_tri = MatrixGenerator.tridiagonal_matrix(
    n=6, diagonal=4.0, off_diagonal=-1.0
)
```

## 📊 Exemplos de Execução

Abaixo estão alguns exemplos de como usar a interface de linha de comando.

### Exemplo 1: Resolver um sistema específico com métodos selecionados

Este comando executa os métodos de Jacobi e Gauss-Seidel no sistema "Sistema Brasileiro 36x36", com uma tolerância de `1e-8` e gera gráficos de convergência.

```bash
python main.py --jacobi --gauss-seidel --system "Sistema Brasileiro 36x36" --tolerance 1e-8 --plots
```

### Exemplo 2: Executar o modo benchmark

O modo benchmark executa uma análise de performance completa em um sistema, testando múltiplas tolerâncias e salvando relatórios detalhados e gráficos.

```bash
python main.py --benchmark --system "Hilbert 36x36" --visualize-benchmark
```

### Exemplo 3: Resolver o sistema não linear

Este comando executa os métodos de Newton, Iteração e Gradiente para resolver o sistema não linear definido no código.

```bash
python main.py --nonlinear --tolerance 1e-7
```

### Exemplo 4: Limpar resultados antigos e executar tudo

Este comando primeiro limpa todos os resultados da pasta `results/` e depois executa todos os métodos em todos os sistemas lineares disponíveis.

```bash
python main.py --clear-old-data --all
```

## 🔬 Testes

Execute os testes para verificar a instalação:

### Com Poetry (Recomendado)

```bash
# Executar todos os testes
poetry run pytest
# ou usando o Makefile
make test

# Com cobertura de código
make test-cov

# Testes rápidos (sem marcador 'slow')
make test-fast

# Testes específicos
poetry run pytest tests/test_iterative_methods.py -v

# Dentro do ambiente virtual ativado
poetry shell
pytest tests/ -v
```

### Com pip/ambiente tradicional

```bash
pytest tests/ -v
pytest tests/ --cov=linear_solver --cov-report=html
```

## 🛠️ Desenvolvimento

Para contribuir com o projeto, consulte o [Guia de Desenvolvimento](DESENVOLVIMENTO.md) que contém:

- Configuração do ambiente de desenvolvimento
- Convenções de código e estilo
- Fluxo de trabalho com Git
- Como adicionar novos métodos
- Execução de testes e verificações de qualidade

### Estrutura do Projeto

A estrutura do projeto foi organizada para separar o código-fonte (`src`) dos testes e da configuração.

```
numerical-methods-for-pde/
├── src/
│   ├── app/
│   │   ├── linear_solver_app.py
│   │   └── nonlinear_solver_app.py
│   ├── analysis/
│   │   └── matrix_analyzer.py
│   ├── benchmark/
│   │   └── main.py
│   ├── cli.py
│   ├── linear_solver/
│   ├── nonlinear_solver/
│   └── utils/
│       └── files.py
├── tests/
├── data/
├── main.py
├── pyproject.toml
├── Makefile
└── README.md
```

## 🤝 Contribuição

Contribuições são bem-vindas! Por favor:

1. Leia o [Guia de Desenvolvimento](DESENVOLVIMENTO.md)
2. Fork o projeto
3. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
4. Siga as convenções de código (use `make format` e `make lint`)
5. Adicione testes para suas mudanças
6. Execute `make check` para validar tudo
7. Commit suas mudanças (`git commit -am 'feat: adiciona nova feature'`)
8. Push para a branch (`git push origin feature/nova-feature`)
9. Abra um Pull Request

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 📚 Referências

- Burden, R. L., & Faires, J. D. (2010). *Numerical Analysis* (9th ed.). Brooks/Cole.
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM.

## 🏷️ Tecnologias Utilizadas

- **Python 3.8+**: Linguagem principal
- **NumPy**: Computação numérica e álgebra linear
- **Matplotlib**: Visualização de resultados
- **Pandas**: Manipulação de dados tabulares  
- **Poetry**: Gerenciamento de dependências e packaging
- **pytest**: Framework de testes
- **Black**: Formatação automática de código
- **mypy**: Verificação de tipos estáticos
- **Pre-commit**: Hooks de qualidade de código

## 👨‍💻 Autor

**João Monteiro**  
📧 joao.monteiro@example.com  
🔗 [GitHub](https://github.com/joaomonteiro)

---

**⭐ Se este projeto foi útil para você, considere dar uma estrela no GitHub!**
