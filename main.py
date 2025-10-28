#!/usr/bin/env python3
"""
Aplicação principal para resolver sistemas lineares e não lineares.
"""
import warnings
from pathlib import Path

# Adiciona o diretório src ao sys.path para permitir importações diretas
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from cli import parse_arguments
from utils.files import (
    clear_old_results,
    descobrir_sistemas_disponiveis,
    carregar_sistema,
    save_solutions,
    create_summary_report
)
from analysis.matrix_analyzer import (
    analyze_matrix_properties,
    analisar_condicionamento_sistema
)
from app.linear_solver_app import (
    solve_with_selected_methods,
    compare_solutions,
    plot_convergence_comparison
)
from app.nonlinear_solver_app import solve_nonlinear_system
from benchmark.main import run_benchmark_mode

# Tenta importar o matplotlib para verificar a disponibilidade
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

warnings.filterwarnings('ignore')

def main():
    """Função principal da aplicação."""
    
    # Parse argumentos da linha de comando
    args = parse_arguments()
    
    title = "BENCHMARK DE TEMPO DOS MÉTODOS" if args.benchmark else "ANÁLISE DE SISTEMAS REAIS"
    print(f"🔬 LINEAR SOLVER - {title}")
    print("=" * 60)
    print(f"⚙️  Modo: {'BENCHMARK' if args.benchmark else 'ANÁLISE NORMAL'}")
    print(f"⚙️  Tolerância: {args.tolerance}")
    print(f"⚙️  Máx. iterações: {args.max_iterations}")
    if not args.benchmark:
        print(f"⚙️  Gráficos: {'Desabilitados' if args.no_plots else 'Habilitados' if HAS_MATPLOTLIB else 'Indisponíveis'}")
        print(f"⚙️  Salvar soluções: {'Sim' if args.save_solutions else 'Não'}")
        print(f"⚙️  Análise condicionamento: {'Desabilitada' if args.skip_conditioning else 'Habilitada'}")
    else:
        print(f"⚙️  Visualizações benchmark: {'Habilitadas' if args.visualize_benchmark and HAS_MATPLOTLIB else 'Desabilitadas' if not args.visualize_benchmark else 'Matplotlib indisponível'}")
    print(f"⚙️  Limpar dados anteriores: {'Sim' if args.clear_old_data else 'Não'}")
    
    # Limpar dados anteriores se solicitado
    if args.clear_old_data:
        clear_old_results()
    
    # Se sistema não linear foi solicitado, resolver e sair
    if args.nonlinear:
        success = solve_nonlinear_system(
            tolerance=args.tolerance,
            max_iterations=args.max_iterations
        )
        if success:
            print("\n🎉 Processamento concluído com sucesso!")
        else:
            print("\n❌ Falha no processamento.")
        return
    
    # Mostrar métodos selecionados (apenas no modo normal)
    if not args.benchmark:
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
        
        print(f"📋 Métodos: {', '.join(selected_methods)}")
    else:
        print(f"📋 Modo benchmark: Todos os métodos aplicáveis serão testados")
    
    # Descobrir sistemas disponíveis na pasta data/
    sistemas = descobrir_sistemas_disponiveis()
    
    if not sistemas:
        print("❌ Nenhum sistema encontrado na pasta data/")
        print("💡 Certifique-se de que há arquivos de dados na pasta ./data/")
        return
    
    # Lista para rastrear sistemas processados com sucesso
    sistemas_processados = []
    
    # Processar cada sistema encontrado
    for i, (nome, tipo, arquivo1, arquivo2) in enumerate(sistemas, 1):
        print(f"\n{'='*60}")
        print(f"PROCESSANDO {i}/{len(sistemas)}: {nome}")
        print(f"{'='*60}")
        
        # Carregar sistema
        A, b = carregar_sistema(nome, tipo, arquivo1, arquivo2)
        if A is None or b is None:
            continue
        
        print(f"✅ Sistema carregado: {A.shape[0]}x{A.shape[1]} + vetor b({b.shape[0]})")
        
        try:
            # Verificar se está no modo benchmark
            if args.benchmark:
                # Executar modo benchmark
                run_benchmark_mode(A, b, nome, args)
                sistemas_processados.append(nome)
            else:
                # Execução normal
                # Analisar propriedades da matriz
                analyze_matrix_properties(A, nome)
                
                # *** ANÁLISE DE CONDICIONAMENTO INTEGRADA ***
                if not args.skip_conditioning:
                    analisar_condicionamento_sistema(A, nome)
                else:
                    print("⚠️  Análise de condicionamento pulada (--skip-conditioning)")
                
                # Resolver com métodos selecionados
                results, solutions = solve_with_selected_methods(A, b, nome, args)
                
                # Comparar soluções obtidas
                compare_solutions(solutions, A, b)
                
                # Salvar soluções se solicitado
                save_solutions(solutions, A, b, nome, args)
                
                # Plotar convergência (apenas para métodos que convergiram)
                if not args.no_plots:
                    converged_results = {name: info for name, info in results.items() 
                                       if info and info['converged']}
                    
                    if converged_results:
                        plot_convergence_comparison(converged_results, nome)
                    else:
                        print("❌ Nenhum método convergiu para plotar")
                
                # Adicionar sistema à lista de processados com sucesso
                sistemas_processados.append(nome)
            
        except Exception as e:
            print(f"💥 Erro ao processar {nome}: {str(e)}")
        
        # Separador entre sistemas
        if i < len(sistemas):
            print("\n" + "~" * 60)
    
    print("\n🎉 ANÁLISE CONCLUÍDA!")
    
    # Criar relatório de execução
    if sistemas_processados:
        create_summary_report(sistemas_processados, args)
    
    # Mostrar estrutura de resultados criada
    results_dir = Path("results")
    if results_dir.exists():
        print(f"📁 Resultados organizados em: {results_dir.absolute()}")
        
        charts_dir = results_dir / "charts"
        if charts_dir.exists() and any(charts_dir.iterdir()):
            chart_count = len(list(charts_dir.glob("*.png")))
            print(f"   📊 Gráficos ({chart_count}): {charts_dir}")
        
        text_results_dir = results_dir / "text_results"
        if text_results_dir.exists() and any(text_results_dir.iterdir()):
            solution_count = len(list(text_results_dir.glob("*.txt")))
            print(f"   💾 Soluções ({solution_count}): {text_results_dir}")
    
    print(f"✅ {len(sistemas_processados)}/{len(sistemas)} sistemas processados com sucesso")


if __name__ == "__main__":
    main()
