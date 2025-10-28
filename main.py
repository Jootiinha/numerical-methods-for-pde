#!/usr/bin/env python3
"""
Aplicação principal para resolver sistemas lineares e não lineares.
"""
import warnings
from pathlib import Path
from src.cli import parse_arguments
from src.utils.files import (
    clear_old_results,
    descobrir_sistemas_disponiveis,
    carregar_sistema,
    save_solutions,
    create_summary_report
)
from src.analysis.matrix_analyzer import (
    analyze_matrix_properties,
    analisar_condicionamento_sistema
)
from src.app.linear_solver_app import (
    solve_with_selected_methods,
    compare_solutions,
    plot_convergence_comparison
)
from src.app.nonlinear_solver_app import solve_nonlinear_system
from src.benchmark.main import run_benchmark_mode

warnings.filterwarnings('ignore')

def _print_execution_summary(args):
    """Imprime um resumo dos parâmetros de execução."""
    title = "BENCHMARK DE TEMPO DOS MÉTODOS" if args.benchmark else "ANÁLISE DE SISTEMAS REAIS"
    print(f"🔬 LINEAR SOLVER - {title}")
    print("=" * 60)
    print(f"⚙️  Modo: {'BENCHMARK' if args.benchmark else 'ANÁLISE NORMAL'}")
    print(f"⚙️  Tolerância: {args.tolerance}")
    print(f"⚙️  Máx. iterações: {args.max_iterations}")
    if not args.benchmark:
        print(f"⚙️  Gráficos: {'Desabilitados' if args.no_plots else 'Habilitados'}")
        print(f"⚙️  Salvar soluções: {'Sim' if args.save_solutions else 'Não'}")
        print(f"⚙️  Análise condicionamento: {'Desabilitada' if args.skip_conditioning else 'Habilitada'}")
    else:
        print(f"⚙️  Visualizações benchmark: {'Habilitadas' if args.visualize_benchmark else 'Desabilitadas'}")
    print(f"⚙️  Limpar dados anteriores: {'Sim' if args.clear_old_data else 'Não'}")
    print("=" * 60)

def _process_linear_system(nome, tipo, arquivo1, arquivo2, args):
    """Carrega e processa um único sistema linear."""
    A, b = carregar_sistema(nome, tipo, arquivo1, arquivo2)
    if A is None or b is None:
        return False

    print(f"✅ Sistema carregado: {A.shape[0]}x{A.shape[1]} + vetor b({b.shape[0]})")

    try:
        if args.benchmark:
            run_benchmark_mode(A, b, nome, args)
        else:
            analyze_matrix_properties(A, nome)
            if not args.skip_conditioning:
                analisar_condicionamento_sistema(A, nome)
            else:
                print("⚠️  Análise de condicionamento pulada (--skip-conditioning)")

            results, solutions = solve_with_selected_methods(A, b, nome, args)
            compare_solutions(solutions, A, b)
            save_solutions(solutions, A, b, nome, args)

            if not args.no_plots:
                converged_results = {name: info for name, info in results.items() if info and info.get('converged')}
                if converged_results:
                    plot_convergence_comparison(converged_results, nome)
                else:
                    print("❌ Nenhum método convergiu para plotar")
        return True
    except Exception as e:
        print(f"💥 Erro ao processar {nome}: {str(e)}")
        return False

def main():
    """Função principal da aplicação."""
    args = parse_arguments()
    _print_execution_summary(args)

    if args.clear_old_data:
        clear_old_results()

    if args.nonlinear:
        if solve_nonlinear_system(tolerance=args.tolerance, max_iterations=args.max_iterations):
            print("\n🎉 Processamento não linear concluído com sucesso!")
        else:
            print("\n❌ Falha no processamento não linear.")
        return

    sistemas = descobrir_sistemas_disponiveis()
    if not sistemas:
        print("❌ Nenhum sistema encontrado na pasta data/")
        print("💡 Certifique-se de que há arquivos de dados na pasta ./data/")
        return

    sistemas_processados = []
    for i, (nome, tipo, arquivo1, arquivo2) in enumerate(sistemas, 1):
        print(f"\n{'='*60}\nPROCESSANDO {i}/{len(sistemas)}: {nome}\n{'='*60}")
        if _process_linear_system(nome, tipo, arquivo1, arquivo2, args):
            sistemas_processados.append(nome)
        if i < len(sistemas):
            print("\n" + "~" * 60)

    print("\n🎉 ANÁLISE CONCLUÍDA!")
    if sistemas_processados:
        create_summary_report(sistemas_processados, args)

    results_dir = Path("results")
    if results_dir.exists():
        print(f"📁 Resultados organizados em: {results_dir.absolute()}")
    
    print(f"✅ {len(sistemas_processados)}/{len(sistemas)} sistemas processados com sucesso")


if __name__ == "__main__":
    main()
