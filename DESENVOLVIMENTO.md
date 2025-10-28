# Guia de Desenvolvimento - Linear Solver

Este guia contém informações para desenvolvedores que desejam contribuir com o projeto.

## 🛠️ Configuração do Ambiente de Desenvolvimento

### Pré-requisitos

- Python 3.8 ou superior
- Poetry (gerenciador de dependências)
- Git

### Instalação do Poetry

```bash
# Linux/macOS
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Via pip (alternativo)
pip install poetry
```

### Configuração Inicial

```bash
# Clonar o repositório
git clone https://github.com/joaomonteiro/numerical-methods-for-pde.git
cd numerical-methods-for-pde

# Instalar todas as dependências
poetry install --with dev,docs,analysis

# Ativar o ambiente virtual
poetry shell

# Instalar hooks de pre-commit (recomendado)
make pre-commit-install
```

## 🏗️ Estrutura do Projeto

A estrutura do projeto foi refatorada para seguir as boas práticas, centralizando todo o código-fonte no diretório `src`.

```
numerical-methods-for-pde/
├── src/
│   ├── app/                        # Lógica da aplicação (orquestração)
│   │   ├── linear_solver_app.py
│   │   └── nonlinear_solver_app.py
│   ├── analysis/                   # Módulos de análise (condicionamento, etc.)
│   │   └── matrix_analyzer.py
│   ├── benchmark/                  # Código para benchmarking de performance
│   │   └── main.py
│   ├── cli.py                      # Definição da interface de linha de comando
│   ├── linear_solver/              # Pacote principal do solver linear
│   ├── nonlinear_solver/           # Pacote principal do solver não linear
│   └── utils/                      # Utilitários gerais
│       └── files.py
├── tests/                          # Testes automatizados
├── data/                           # Dados de entrada (matrizes, vetores)
├── main.py                         # Ponto de entrada da aplicação
├── pyproject.toml                  # Configuração do projeto e dependências
├── Makefile                        # Comandos de automação
└── DESENVOLVIMENTO.md              # Guia para desenvolvedores
```

## 🧪 Executando Testes

### Comandos Básicos

```bash
# Todos os testes
make test

# Testes com cobertura
make test-cov

# Apenas testes rápidos
make test-fast

# Teste específico
poetry run pytest tests/test_jacobi.py -v
```

### Marcadores de Teste

- `unit`: Testes unitários
- `integration`: Testes de integração
- `slow`: Testes que demoram mais para executar
- `convergence`: Testes específicos de convergência

```bash
# Executar apenas testes unitários
poetry run pytest -m unit

# Pular testes lentos
poetry run pytest -m "not slow"
```

## 🎨 Qualidade de Código

### Formatação

```bash
# Formatar código automaticamente
make format

# Verificar formatação sem modificar
make format-check
```

### Linting

```bash
# Executar verificações de código
make lint

# Verificações completas
make check  # format-check + lint + test
```

### Ferramentas Configuradas

- **Black**: Formatação automática de código
- **isort**: Organização de imports
- **flake8**: Verificações de estilo e qualidade
- **mypy**: Verificação de tipos estáticos
- **pre-commit**: Hooks automáticos de verificação

## 📝 Convenções de Código

### Estilo

- Seguimos PEP 8 com formatação Black (linha max: 88 caracteres)
- Imports organizados por isort
- Type hints obrigatórios em funções públicas
- Docstrings no formato Google/NumPy

### Exemplo de Função

```python
def solve_system(A: np.ndarray, b: np.ndarray, 
                tolerance: float = 1e-6) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Resolve um sistema linear usando método iterativo.
    
    Args:
        A: Matriz de coeficientes (n x n)
        b: Vetor de termos independentes (n,)
        tolerance: Tolerância para convergência
        
    Returns:
        Tupla contendo:
        - Solução do sistema
        - Informações de convergência
        
    Raises:
        ValueError: Se as dimensões forem incompatíveis
    """
    # Implementação aqui
    pass
```

### Nomeação

- **Variáveis e funções**: `snake_case`
- **Classes**: `PascalCase`
- **Constantes**: `UPPER_SNAKE_CASE`
- **Arquivos e módulos**: `snake_case`

## 🔄 Fluxo de Desenvolvimento

### 1. Criando uma Nova Feature

```bash
# Criar branch para a feature
git checkout -b feature/nova-funcionalidade

# Fazer alterações
# ...

# Executar verificações
make check

# Commit e push
git add .
git commit -m "feat: adiciona nova funcionalidade"
git push origin feature/nova-funcionalidade
```

### 2. Adicionando um Novo Método

1. Criar arquivo na pasta `src/linear_solver/methods/`
2. Herdar de `LinearSolver`
3. Implementar métodos abstratos
4. Adicionar ao `__init__.py` do módulo methods
5. Criar testes correspondentes
6. Atualizar documentação

### 3. Exemplo: Novo Método SOR

```python
# src/linear_solver/methods/sor.py
from typing import Tuple, Optional, Dict, Any
import numpy as np
from ..base import LinearSolver

class SORSolver(LinearSolver):
    """Método SOR (Successive Over-Relaxation)."""
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 1000,
                 omega: float = 1.5):
        super().__init__(tolerance, max_iterations)
        self.omega = omega
        
    def get_method_name(self) -> str:
        return f"SOR (ω={self.omega:.2f})"
        
    def solve(self, A: np.ndarray, b: np.ndarray, 
              x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Implementação do método SOR
        pass
```

## 🧩 Adicionando Dependências

### Dependência Principal

```bash
poetry add numpy>=1.21.0
```

### Dependência de Desenvolvimento

```bash
poetry add --group dev pytest-benchmark
```

### Dependência Opcional

```bash
poetry add --group analysis plotly
```

## 🏷️ Versionamento

Seguimos [Semantic Versioning](https://semver.org/):

- **MAJOR**: Mudanças incompatíveis na API
- **MINOR**: Funcionalidades adicionadas (compatível)
- **PATCH**: Correções de bugs (compatível)

```bash
# Incrementar versão
make bump-patch  # 1.0.0 -> 1.0.1
make bump-minor  # 1.0.0 -> 1.1.0
make bump-major  # 1.0.0 -> 2.0.0
```

## 📦 Build e Publicação

### Build Local

```bash
make build
```

### Publicação (PyPI)

```bash
# Testar no TestPyPI primeiro
make publish-test

# Publicar no PyPI oficial
make publish
```

## 🐛 Debug e Troubleshooting

### Problemas Comuns

1. **Erro de import**: Verificar se está no ambiente virtual poetry
2. **Testes falhando**: Executar `poetry install --with dev`
3. **Formatação incorreta**: Executar `make format`

### Comandos Úteis

```bash
# Informações do ambiente
make info

# Dependências desatualizadas
make show-outdated

# Limpar cache
make clean
```

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature
3. Implemente seguindo as convenções
4. Adicione testes
5. Execute `make check`
6. Faça commit com mensagem clara
7. Abra Pull Request

### Mensagens de Commit

Seguimos [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` Nova funcionalidade
- `fix:` Correção de bug
- `docs:` Atualização de documentação
- `test:` Adição/modificação de testes
- `refactor:` Refatoração sem mudança de funcionalidade
- `perf:` Melhorias de performance
- `chore:` Atualizações de dependências, configuração, etc.

## 📞 Suporte

- **Issues**: [GitHub Issues](https://github.com/joaomonteiro/numerical-methods-for-pde/issues)
- **Discussões**: [GitHub Discussions](https://github.com/joaomonteiro/numerical-methods-for-pde/discussions)
- **Email**: joao.monteiro@example.com

## 📚 Recursos Adicionais

- [Poetry Documentation](https://python-poetry.org/docs/)
- [pytest Documentation](https://docs.pytest.org/)
- [Black Code Style](https://black.readthedocs.io/)
- [NumPy Documentation](https://numpy.org/doc/)
