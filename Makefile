# Makefile para automatizar tarefas comuns do projeto

.PHONY: help install install-dev test lint format clean build publish docs

# Variáveis
POETRY := poetry
PYTHON := $(POETRY) run python
PYTEST := $(POETRY) run pytest
BLACK := $(POETRY) run black
ISORT := $(POETRY) run isort
FLAKE8 := $(POETRY) run flake8
MYPY := $(POETRY) run mypy

help: ## Mostrar esta ajuda
	@echo "Comandos disponíveis:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Instalar dependências básicas
	$(POETRY) install

install-dev: ## Instalar com dependências de desenvolvimento
	$(POETRY) install --with dev,docs,analysis

install-all: ## Instalar todas as dependências
	$(POETRY) install --with dev,docs,analysis

test: ## Executar todos os testes
	$(PYTEST) tests/ -v

test-cov: ## Executar testes com cobertura
	$(PYTEST) tests/ --cov=linear_solver --cov-report=html --cov-report=term

test-fast: ## Executar testes rápidos (sem marcador 'slow')
	$(PYTEST) tests/ -v -m "not slow"

lint: ## Executar verificações de código (flake8, mypy)
	$(FLAKE8) linear_solver/ tests/
	$(MYPY) linear_solver/

format: ## Formatar código (black + isort)
	$(BLACK) linear_solver/ tests/ main.py
	$(ISORT) linear_solver/ tests/ main.py

format-check: ## Verificar formatação sem modificar
	$(BLACK) --check linear_solver/ tests/ main.py
	$(ISORT) --check-only linear_solver/ tests/ main.py

clean: ## Limpar arquivos temporários
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/

build: ## Construir pacote
	$(POETRY) build

publish: ## Publicar no PyPI (requer configuração)
	$(POETRY) publish

publish-test: ## Publicar no TestPyPI
	$(POETRY) publish --repository testpypi

docs: ## Gerar documentação (se configurada)
	@echo "Documentação será implementada futuramente"

run-example: ## Executar exemplo principal
	$(PYTHON) main.py

shell: ## Ativar ambiente virtual Poetry
	$(POETRY) shell

update: ## Atualizar dependências
	$(POETRY) update

check: format-check lint test ## Executar todas as verificações

pre-commit-install: ## Instalar hooks de pre-commit
	$(POETRY) run pre-commit install

pre-commit-run: ## Executar pre-commit em todos os arquivos
	$(POETRY) run pre-commit run --all-files

# Comandos para CI/CD
ci-install: ## Instalar dependências para CI
	$(POETRY) install --with dev

ci-test: ## Executar testes para CI
	$(PYTEST) tests/ --cov=linear_solver --cov-report=xml

ci-lint: ## Verificações para CI
	$(FLAKE8) linear_solver/ tests/
	$(BLACK) --check linear_solver/ tests/ main.py
	$(ISORT) --check-only linear_solver/ tests/ main.py

# Utilitários
show-deps: ## Mostrar árvore de dependências
	$(POETRY) show --tree

show-outdated: ## Mostrar dependências desatualizadas
	$(POETRY) show --outdated

info: ## Informações do projeto
	$(POETRY) env info
	$(POETRY) show

version: ## Mostrar versão atual
	$(POETRY) version

bump-patch: ## Incrementar versão patch (1.0.0 -> 1.0.1)
	$(POETRY) version patch

bump-minor: ## Incrementar versão minor (1.0.0 -> 1.1.0)
	$(POETRY) version minor

bump-major: ## Incrementar versão major (1.0.0 -> 2.0.0)
	$(POETRY) version major
