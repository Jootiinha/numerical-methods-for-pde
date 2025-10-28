import numpy as np
from pathlib import Path
from linear_solver import (
    JacobiSolver, GaussSeidelSolver, ConjugateGradientSolver,
    JacobiOrder2Solver, GaussSeidelOrder2Solver,
    PreconditionedConjugateGradientSolver,
    MatrixValidator
)

# Tenta importar o matplotlib para verificar a disponibilidade
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def build_methods_list(args, analysis):
    """Constrói a lista de métodos baseada nos argumentos fornecidos."""
    
    methods = []
    
    # Métodos básicos
    if args.all or args.jacobi:
        methods.append(("Jacobi", JacobiSolver(
            tolerance=args.tolerance, 
            max_iterations=args.max_iterations
        )))
    
    if args.all or args.gauss_seidel:
        methods.append(("Gauss-Seidel", GaussSeidelSolver(
            tolerance=args.tolerance, 
            max_iterations=args.max_iterations
        )))
    
    # Métodos de segunda ordem
    if args.all or args.jacobi_order2:
        methods.append(("Jacobi Ordem 2", JacobiOrder2Solver(
            tolerance=args.tolerance, 
            max_iterations=args.max_iterations,
            omega1=0.7, omega2=0.2, omega3=0.1
        )))
    
    if args.all or args.gauss_seidel_order2:
        methods.append(("Gauss-Seidel Ordem 2", GaussSeidelOrder2Solver(
            tolerance=args.tolerance, 
            max_iterations=args.max_iterations,
            relaxation_factor=1.2, omega1=0.8, omega2=0.15, omega3=0.05
        )))
    
    # Métodos para matrizes simétricas positivas definidas
    if analysis['is_symmetric'] and analysis['is_positive_definite']:
        if args.all or args.conjugate_gradient:
            methods.append(("Gradiente Conjugado", ConjugateGradientSolver(
                tolerance=args.tolerance
            )))
        
        if args.all or args.preconditioned_cg:
            methods.append(("Gradiente Conjugado Precondicionado", 
                           PreconditionedConjugateGradientSolver(
                               tolerance=args.tolerance
                           )))
    else:
        # Avisar se métodos específicos para SPD foram solicitados mas a matriz não é SPD
        if args.conjugate_gradient or args.preconditioned_cg:
            print("⚠️  Métodos de Gradiente Conjugado ignorados: matriz não é simétrica positiva definida")
    
    return methods


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


def plot_convergence_comparison(results, matrix_name):
    """Plota comparação de convergência dos métodos."""
    
    if not HAS_MATPLOTLIB:
        print("⚠️  Matplotlib não disponível - pulando gráficos")
        return
    
    plt.figure(figsize=(14, 10))
    
    # Subplot 1: Histórico de convergência  
    plt.subplot(2, 2, 1)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for i, (method_name, info) in enumerate(results.items()):
        if info and info['converged'] and 'convergence_history' in info:
            history = info['convergence_history']
            if history:
                plt.semilogy(
                    range(1, len(history) + 1), history, 
                    label=method_name, 
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)],
                    markersize=4, linewidth=2
                )
    
    plt.xlabel('Iteração')
    plt.ylabel('Erro (escala log)')
    plt.title(f'Convergência - {matrix_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Número de iterações
    plt.subplot(2, 2, 2)
    
    method_names = []
    iterations = []
    colors_bar = []
    
    for i, (method_name, info) in enumerate(results.items()):
        if info and info['converged']:
            method_names.append(method_name)
            iterations.append(info['iterations'])
            colors_bar.append(colors[i % len(colors)])
    
    if method_names:
        bars = plt.bar(method_names, iterations, color=colors_bar, alpha=0.7)
        plt.ylabel('Número de Iterações')
        plt.title('Iterações para Convergência')
        plt.xticks(rotation=45, ha='right')
        
        # Adicionar valores nas barras
        for bar, iter_count in zip(bars, iterations):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(iter_count), ha='center', va='bottom')
    
    # Subplot 3: Erro final
    plt.subplot(2, 2, 3)
    
    method_names_error = []
    final_errors = []
    colors_error = []
    
    for i, (method_name, info) in enumerate(results.items()):
        if info and info['converged']:
            method_names_error.append(method_name)
            final_errors.append(info['final_error'])
            colors_error.append(colors[i % len(colors)])
    
    if method_names_error:
        bars = plt.bar(method_names_error, final_errors, color=colors_error, alpha=0.7)
        plt.ylabel('Erro Final')
        plt.yscale('log')
        plt.title('Erro Final por Método')
        plt.xticks(rotation=45, ha='right')
    
    # Subplot 4: Comparação de resíduos (se disponível)
    plt.subplot(2, 2, 4)
    
    if any('residual_history' in info for info in results.values() if info):
        for i, (method_name, info) in enumerate(results.items()):
            if info and 'residual_history' in info and info['residual_history']:
                plt.semilogy(
                    range(1, len(info['residual_history']) + 1), 
                    info['residual_history'],
                    label=method_name,
                    color=colors[i % len(colors)],
                    marker=markers[i % len(markers)],
                    markersize=4, linewidth=2
                )
        plt.xlabel('Iteração')
        plt.ylabel('Resíduo (escala log)')
        plt.title('História do Resíduo')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Dados de resíduo\nnão disponíveis', 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    
    # Criar estrutura de diretórios para gráficos
    results_dir = Path("results")
    charts_dir = results_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    # Salvar gráfico na pasta apropriada
    chart_filename = f'convergencia_{matrix_name.replace(" ", "_").lower()}.png'
    chart_path = charts_dir / chart_filename
    
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo: {chart_path}")
    plt.show()
