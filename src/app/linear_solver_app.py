import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.linear_solver.methods import (
    JacobiSolver, GaussSeidelSolver, ConjugateGradientSolver,
    PreconditionedConjugateGradientSolver
)
from src.linear_solver.utils.matrix_validator import MatrixValidator

def build_methods_list(args, analysis):
    """Constrói a lista de métodos com base nos argumentos e análise da matriz."""
    
    method_configs = {
        'jacobi': (JacobiSolver, {}),
        'gauss_seidel': (GaussSeidelSolver, {}),
        'sor': (GaussSeidelSolver, {'relaxation_factor': 1.25}), # Exemplo de SOR
        'conjugate_gradient': (ConjugateGradientSolver, {}),
        'preconditioned_cg': (PreconditionedConjugateGradientSolver, {})
    }
    
    methods_to_run = []
    if args.all:
        methods_to_run.extend(method_configs.keys())
    else:
        for name in method_configs:
            if getattr(args, name, False):
                methods_to_run.append(name)
    
    # Adicionar SOR se Gauss-Seidel for selecionado
    if 'gauss_seidel' in methods_to_run and 'sor' not in methods_to_run:
        methods_to_run.append('sor')

    # Filtrar métodos de GC para matrizes não-SPD
    is_spd = analysis.get('is_symmetric', False) and analysis.get('is_positive_definite', False)
    if not is_spd:
        spd_methods = {'conjugate_gradient', 'preconditioned_cg'}
        if any(m in methods_to_run for m in spd_methods):
             print("⚠️  Métodos de Gradiente Conjugado ignorados: matriz não é simétrica positiva definida")
        methods_to_run = [m for m in methods_to_run if m not in spd_methods]

    # Construir instâncias dos solvers
    solvers = []
    for name in methods_to_run:
        SolverClass, params = method_configs[name]
        instance = SolverClass(
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
            **params
        )
        solvers.append((instance.get_method_name(), instance))
        
    return solvers


def solve_with_selected_methods(A, b, matrix_name, args):
    """Resolve o sistema com os métodos selecionados."""
    
    print(f"\n🔧 RESOLVENDO SISTEMA: {matrix_name}")
    print("-" * 50)
    
    # Analisar matriz para determinar métodos aplicáveis
    analysis = MatrixValidator.analyze_matrix(A)
    
    # Construir lista de métodos baseada nos argumentos
    methods = build_methods_list(args, analysis)
    
    if not methods:
        print("❌ Nenhum método aplicável selecionado!")
        return {}, {}
    
    print(f"📋 Métodos selecionados: {', '.join([name for name, _ in methods])}")
    
    results = {}
    solutions = {}
    
    for method_name, solver in methods:
        try:
            print(f"\n⚡ {method_name}:")
            
            x, info = solver.solve(A, b)
            results[method_name] = info
            solutions[method_name] = x
            
            if info['converged']:
                print(f"   ✅ Convergiu em {info['iterations']} iterações")
                print(f"   📊 Erro final: {info['final_error']:.2e}")
                
                # Verificar qualidade da solução
                residual = np.linalg.norm(A @ x - b)
                print(f"   🎯 Resíduo: {residual:.2e}")
                
                # Mostrar algumas componentes da solução
                print(f"   🔢 x[0:5]: {x[:5] if len(x) >= 5 else x}")
                if len(x) > 5:
                    print(f"   🔢 x[-5:]: {x[-5:]}")
            else:
                print(f"   ❌ Não convergiu após {info['iterations']} iterações")
                print(f"   📊 Erro final: {info['final_error']:.2e}")
                
        except Exception as e:
            print(f"   💥 Erro: {str(e)}")
            results[method_name] = None
    
    return results, solutions


def compare_solutions(solutions, A, b):
    """Compara as soluções obtidas pelos diferentes métodos."""
    
    print("\n🔍 COMPARAÇÃO DAS SOLUÇÕES")
    print("-" * 50)
    
    if not solutions:
        print("Nenhuma solução disponível para comparação.")
        return
    
    # Calcular solução de referência (numpy)
    try:
        x_ref = np.linalg.solve(A, b)
        print(f"Solução de referência (NumPy): {x_ref}")
        
        print("\nComparação com solução de referência:")
        for method_name, x in solutions.items():
            if x is not None:
                error = np.linalg.norm(x - x_ref, ord=np.inf)
                print(f"  {method_name:20}: erro = {error:.2e}")
                
    except np.linalg.LinAlgError:
        print("Não foi possível calcular solução de referência (matriz singular)")
        
        # Comparar soluções entre si
        method_names = list(solutions.keys())
        if len(method_names) > 1:
            print("\nComparação entre métodos:")
            ref_method = method_names[0]
            x_ref = solutions[ref_method]
            
            for method_name in method_names[1:]:
                x = solutions[method_name]
                if x is not None and x_ref is not None:
                    error = np.linalg.norm(x - x_ref, ord=np.inf)
                    print(f"  {method_name} vs {ref_method}: erro = {error:.2e}")


def _plot_subplot(ax, data, plot_type, title, xlabel, ylabel, use_log=False):
    """Função auxiliar para criar um subplot."""
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    
    if plot_type == 'line':
        for i, (name, values) in enumerate(data.items()):
            if values:
                ax.semilogy(range(1, len(values) + 1), values, label=name, color=colors[i], marker='o', markersize=4, linewidth=2)
        ax.legend()
    elif plot_type == 'bar':
        names = list(data.keys())
        values = list(data.values())
        bars = ax.bar(names, values, color=colors, alpha=0.8)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2e}' if use_log else int(yval), va='bottom', ha='center')
        ax.set_xticklabels(names, rotation=45, ha="right")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if use_log:
        ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", alpha=0.3)

def plot_convergence_comparison(results, matrix_name):
    """Plota comparação de convergência dos métodos de forma modular."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Análise de Convergência para '{matrix_name}'", fontsize=16)
    
    converged_results = {name: info for name, info in results.items() if info and info.get('converged')}
    
    # Dados para os plots
    error_history = {name: info['convergence_history'] for name, info in converged_results.items()}
    residual_history = {name: info['residual_history'] for name, info in converged_results.items() if 'residual_history' in info}
    iterations = {name: info['iterations'] for name, info in converged_results.items()}
    final_errors = {name: info['final_error'] for name, info in converged_results.items()}

    # Criar subplots
    _plot_subplot(axes[0, 0], error_history, 'line', 'Histórico de Erro', 'Iteração', 'Erro')
    _plot_subplot(axes[0, 1], residual_history, 'line', 'Histórico de Resíduo', 'Iteração', 'Resíduo')
    _plot_subplot(axes[1, 0], iterations, 'bar', 'Iterações até Convergência', 'Método', 'Número de Iterações')
    _plot_subplot(axes[1, 1], final_errors, 'bar', 'Erro Final', 'Método', 'Erro Final', use_log=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Salvar figura
    charts_dir = Path("results") / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_path = charts_dir / f'convergencia_{matrix_name.replace(" ", "_").lower()}.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo: {chart_path}")
    plt.show()
