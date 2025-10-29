import numpy as np
import yaml
from pathlib import Path
from src.linear_solver.utils.csv_loader import CSVMatrixLoader

def load_config():
    """Carrega as configurações do arquivo config.yaml."""
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("⚠️  Arquivo config.yaml não encontrado. Usando valores padrão.")
        return {
            'decimal_places': 4,
            'suppress_scientific_notation': True
        }
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            # Retorna o dicionário 'settings' ou um dicionário vazio se não for encontrado
            return config.get('settings', {
                'decimal_places': 4,
                'suppress_scientific_notation': True
            })
    except Exception as e:
        print(f"❌ Erro ao ler config.yaml: {e}. Usando valores padrão.")
        return {
            'decimal_places': 4,
            'suppress_scientific_notation': True
        }

def clear_old_results():
    """Limpa resultados anteriores das pastas de output."""
    
    print("\n🗑️  LIMPANDO RESULTADOS ANTERIORES")
    print("-" * 50)
    
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("📁 Pasta 'results/' não existe - nada para limpar")
        return
    
    # Listar arquivos/pastas que serão removidos
    items_to_remove = []
    
    # Arquivos na raiz de results/
    for item in results_dir.iterdir():
        if item.is_file():
            items_to_remove.append(("arquivo", item))
    
    # Subpastas específicas
    subdirs_to_clear = ["text_results", "charts"]
    for subdir_name in subdirs_to_clear:
        subdir = results_dir / subdir_name
        if subdir.exists() and subdir.is_dir():
            for item in subdir.iterdir():
                items_to_remove.append(("arquivo", item))
            items_to_remove.append(("pasta", subdir))
    
    if not items_to_remove:
        print("✅ Nenhum resultado anterior encontrado")
        return
    
    # Mostrar o que será removido
    print(f"📋 Itens a serem removidos ({len(items_to_remove)}):")
    
    files_count = 0
    dirs_count = 0
    
    for item_type, item_path in items_to_remove:
        # Usar o caminho como está (já é relativo ao diretório atual)
        display_path = str(item_path)
        
        if item_type == "arquivo":
            files_count += 1
            print(f"   📄 {display_path}")
        elif item_type == "pasta":
            dirs_count += 1
            print(f"   📁 {display_path}/")
    
    # Remover os itens
    removed_files = 0
    removed_dirs = 0
    errors = []
    
    for item_type, item_path in items_to_remove:
        try:
            if item_type == "arquivo" and item_path.exists():
                item_path.unlink()
                removed_files += 1
            elif item_type == "pasta" and item_path.exists():
                # Só remove se estiver vazia (arquivos já foram removidos)
                if not any(item_path.iterdir()):
                    item_path.rmdir()
                    removed_dirs += 1
        except Exception as e:
            errors.append(f"{item_path}: {str(e)}")
    
    # Relatório final
    print(f"\n✅ Limpeza concluída:")
    print(f"   📄 Arquivos removidos: {removed_files}")
    print(f"   📁 Pastas removidas: {removed_dirs}")
    
    if errors:
        print(f"   ⚠️  Erros ({len(errors)}):")
        for error in errors:
            print(f"      {error}")
    
    print("🆕 Pronto para novos resultados!")


def carregar_matriz_brasileira(filepath):
    """Carrega matriz do formato brasileiro (vírgulas decimais, tabs)."""
    filepath = Path(filepath)
    if not filepath.exists():
        return None
        
    matriz = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for linha in file:
            linha = linha.strip()
            if not linha:
                continue
                
            valores = linha.split('\t')
            linha_numerica = []
            
            for valor in valores:
                valor = valor.strip()
                if valor:
                    valor_num = float(valor.replace(',', '.'))
                    linha_numerica.append(valor_num)
            
            if linha_numerica:
                matriz.append(linha_numerica)
    
    return np.array(matriz)


def carregar_vetor_brasileiro(filepath):
    """Carrega vetor do formato brasileiro (vírgulas decimais)."""
    filepath = Path(filepath)
    if not filepath.exists():
        return None
        
    vetor = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for linha in file:
            linha = linha.strip()
            if not linha:
                continue
                
            valor = float(linha.replace(',', '.'))
            vetor.append(valor)
    
    return np.array(vetor)


def descobrir_sistemas_disponiveis():
    """Descobre todos os sistemas lineares disponíveis na pasta data/."""
    
    print("=" * 60)
    print("DESCOBRINDO SISTEMAS NA PASTA ./data/")
    print("=" * 60)
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Pasta 'data/' não encontrada!")
        return []
    
    sistemas = []
    
    # 1. Sistema brasileiro específico (Matriz_A.txt + Vetor_b.txt)
    matriz_a_path = data_dir / "Matriz_A.txt"
    vetor_b_path = data_dir / "Vetor_b.txt"
    
    if matriz_a_path.exists() and vetor_b_path.exists():
        print("✅ Sistema brasileiro encontrado: Matriz_A.txt + Vetor_b.txt")
        sistemas.append(("Sistema Brasileiro 36x36", "brasileiro", matriz_a_path, vetor_b_path))
    
    # 2. Arquivos CSV de matriz aumentada
    for csv_file in data_dir.glob("*.csv"):
        if "36x36" in csv_file.name and not csv_file.name.startswith("matriz_A_") and not csv_file.name.startswith("vetor_b_"):
            nome = csv_file.stem.replace("_36x36", "").replace("_", " ").title()
            print(f"✅ Sistema CSV encontrado: {csv_file.name}")
            sistemas.append((nome, "csv_aumentado", csv_file, None))
    
    # 3. Pares de arquivos CSV separados (matriz_A_* + vetor_b_*)
    matriz_files = list(data_dir.glob("matriz_A_*.csv"))
    for matriz_file in matriz_files:
        nome_base = matriz_file.name.replace("matriz_A_", "").replace(".csv", "")
        vetor_file = data_dir / f"vetor_b_{nome_base}.csv"
        
        if vetor_file.exists():
            nome = nome_base.replace("_36x36", "").replace("_", " ").title()
            print(f"✅ Sistema CSV separado encontrado: {matriz_file.name} + {vetor_file.name}")
            sistemas.append((nome + " (Separado)", "csv_separado", matriz_file, vetor_file))
    
    print(f"\n📊 Total de sistemas encontrados: {len(sistemas)}")
    return sistemas

def carregar_sistema(nome, tipo, arquivo1, arquivo2=None):
    """Carrega um sistema linear baseado no tipo especificado."""
    
    try:
        if tipo == "brasileiro":
            A = carregar_matriz_brasileira(arquivo1)
            b = carregar_vetor_brasileiro(arquivo2)
            if A is None or b is None:
                raise ValueError("Falha ao carregar arquivos brasileiros")
            
        elif tipo == "csv_aumentado":
            A, b = CSVMatrixLoader.load_augmented_matrix(arquivo1, skip_header=True)
            
        elif tipo == "csv_separado":
            A, b = CSVMatrixLoader.load_separate_files(arquivo1, arquivo2)
            
        else:
            raise ValueError(f"Tipo de sistema desconhecido: {tipo}")
        
        # Verificar dimensões
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Matriz A não é quadrada: {A.shape}")
        
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"Dimensões incompatíveis: A {A.shape}, b {b.shape}")
        
        return A, b
        
    except Exception as e:
        print(f"❌ Erro ao carregar {nome}: {str(e)}")
        return None, None

def save_solutions(solutions, A, b, matrix_name, args):
    """Salva as soluções em arquivos se solicitado."""
    
    if not args.save_solutions:
        return
    
    if not solutions:
        print("❌ Nenhuma solução para salvar")
        return
    
    # Criar estrutura de diretórios
    results_dir = Path("results")
    text_results_dir = results_dir / "text_results"
    text_results_dir.mkdir(parents=True, exist_ok=True)
    
    nome_arquivo_base = matrix_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    
    # Calcular solução de referência
    try:
        x_ref = np.linalg.solve(A, b)
        
        # Salvar solução de referência
        arquivo_ref = text_results_dir / f"solucao_{nome_arquivo_base}_referencia.txt"
        with open(arquivo_ref, 'w', encoding='utf-8') as f:
            f.write(f"# Solução de referência: {matrix_name}\n")
            f.write(f"# Método: NumPy (direto)\n")
            f.write(f"# Data: {Path().absolute()}\n")
            f.write("#" + "="*50 + "\n")
            
            for i, valor in enumerate(x_ref):
                f.write(f"{i+1:3d}  {valor:15.10f}\n")
        
        print(f"✅ Solução de referência salva: {arquivo_ref}")
        
        # Salvar soluções dos métodos iterativos
        for method_name, x in solutions.items():
            if x is not None:
                method_filename = method_name.lower().replace(" ", "_").replace("-", "_")
                arquivo_metodo = text_results_dir / f"solucao_{nome_arquivo_base}_{method_filename}.txt"
                
                error = np.linalg.norm(x - x_ref)
                residual = np.linalg.norm(A @ x - b)
                
                with open(arquivo_metodo, 'w', encoding='utf-8') as f:
                    f.write(f"# Solução: {matrix_name}\n")
                    f.write(f"# Método: {method_name}\n")
                    f.write(f"# Erro vs referência: {error:.2e}\n")
                    f.write(f"# Resíduo: {residual:.2e}\n")
                    f.write(f"# Tolerância usada: {args.tolerance}\n")
                    f.write(f"# Máx. iterações: {args.max_iterations}\n")
                    f.write("#" + "="*50 + "\n")
                    
                    for i, valor in enumerate(x):
                        f.write(f"{i+1:3d}  {valor:15.10f}\n")
                
                print(f"✅ Solução {method_name} salva: {arquivo_metodo.name}")
                
    except np.linalg.LinAlgError as e:
        print(f"❌ Não foi possível salvar soluções: {e}")
    
    print(f"📁 Arquivos salvos em: {text_results_dir.absolute()}")

def create_summary_report(sistemas_processados, args):
    """Cria um relatório resumo da execução."""
    
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = results_dir / "summary_report.txt"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("LINEAR SOLVER - RELATÓRIO DE EXECUÇÃO\n")
        f.write("=" * 60 + "\n\n")
        
        # Configurações usadas
        f.write("CONFIGURAÇÕES DA EXECUÇÃO:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Tolerância: {args.tolerance}\n")
        f.write(f"Máximo de iterações: {args.max_iterations}\n")
        f.write(f"Gráficos habilitados: {'Não' if args.no_plots else 'Sim'}\n")
        f.write(f"Salvar soluções: {'Sim' if args.save_solutions else 'Não'}\n")
        f.write(f"Análise de condicionamento: {'Desabilitada' if args.skip_conditioning else 'Habilitada'}\n")
        f.write(f"Dados anteriores limpos: {'Sim' if args.clear_old_data else 'Não'}\n")
        
        # Métodos selecionados
        selected_methods = []
        if args.all:
            selected_methods.append("TODOS")
        else:
            if args.jacobi: selected_methods.append("Jacobi")
            if args.gauss_seidel: selected_methods.append("Gauss-Seidel")
            if args.conjugate_gradient: selected_methods.append("Gradiente Conjugado")
            if args.jacobi_order2: selected_methods.append("Jacobi Ordem 2")
            if args.gauss_seidel_order2: selected_methods.append("Gauss-Seidel Ordem 2")
            if args.preconditioned_cg: selected_methods.append("Gradiente Conjugado Precondicionado")
        
        f.write(f"Métodos selecionados: {', '.join(selected_methods)}\n\n")
        
        # Sistemas processados
        f.write("SISTEMAS PROCESSADOS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total de sistemas encontrados: {len(sistemas_processados)}\n\n")
        
        for i, sistema in enumerate(sistemas_processados, 1):
            f.write(f"{i:2d}. {sistema}\n")
        
        f.write(f"\nRelatório gerado em: {Path().absolute()}\n")
    
    print(f"📋 Relatório de execução salvo: {summary_file}")
