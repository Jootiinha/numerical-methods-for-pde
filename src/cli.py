import argparse

def parse_arguments():
    """Parse argumentos da linha de comando."""
    
    parser = argparse.ArgumentParser(
        description="Resolver sistemas lineares usando métodos iterativos",
        epilog="""
            Exemplos:
                %(prog)s --all                      # Todos os métodos disponíveis
                %(prog)s --jacobi                   # Apenas método de Jacobi
                %(prog)s --jacobi --gauss-seidel    # Jacobi e Gauss-Seidel
                %(prog)s --conjugate-gradient       # Apenas Gradiente Conjugado
                %(prog)s --jacobi-order2            # Jacobi de segunda ordem
                %(prog)s --no-plots                 # Executar sem gráficos
                %(prog)s --clear-old-data           # Limpar resultados anteriores
                %(prog)s --all --save-solutions --clear-old-data # Execução completa limpa
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Métodos disponíveis
    parser.add_argument('--all', action='store_true',
                       help='Executar todos os métodos disponíveis')
    
    parser.add_argument('--jacobi', action='store_true',
                       help='Executar método de Jacobi')
    
    parser.add_argument('--gauss-seidel', action='store_true',
                       help='Executar método de Gauss-Seidel')
    
    parser.add_argument('--conjugate-gradient', action='store_true',
                       help='Executar Gradiente Conjugado (apenas para matrizes simétricas positivas definidas)')
    
    parser.add_argument('--jacobi-order2', action='store_true',
                       help='Executar Jacobi de segunda ordem')
    
    parser.add_argument('--gauss-seidel-order2', action='store_true',
                       help='Executar Gauss-Seidel de segunda ordem')
    
    parser.add_argument('--preconditioned-cg', action='store_true',
                       help='Executar Gradiente Conjugado Precondicionado')
    
    # Opções de execução
    parser.add_argument('--no-plots', action='store_true',
                       help='Não gerar gráficos (útil se matplotlib não estiver disponível)')
    
    parser.add_argument('--tolerance', type=float, default=1e-4,
                       help='Tolerância para convergência (padrão: 1e-4 = 10^(-4))')
    
    parser.add_argument('--max-iterations', type=int, default=5000,
                       help='Número máximo de iterações (padrão: 5000)')
    
    parser.add_argument('--save-solutions', action='store_true',
                       help='Salvar soluções em arquivos')
    
    parser.add_argument('--skip-conditioning', action='store_true',
                       help='Pular análise de condicionamento (mais rápido)')
    
    parser.add_argument('--clear-old-data', action='store_true',
                       help='Limpar resultados anteriores antes de executar')
    
    parser.add_argument('--benchmark', action='store_true',
                       help='Executar benchmark de tempo dos métodos (múltiplas rodadas para estatísticas)')
    
    parser.add_argument('--visualize-benchmark', action='store_true',
                       help='Gerar visualizações avançadas do benchmark (requer --benchmark)')
    
    # Sistemas não lineares
    parser.add_argument('--nonlinear', action='store_true',
                       help='Resolver sistema não linear específico usando Newton, Iteração e Gradiente')
    
    args = parser.parse_args()
    
    # Validar dependências de argumentos
    if args.visualize_benchmark and not args.benchmark:
        parser.error("--visualize-benchmark requer --benchmark")
    
    # Se nenhum método foi especificado E não é sistema não linear, usar --all
    if not args.nonlinear and not any([args.all, args.jacobi, args.gauss_seidel, args.conjugate_gradient,
                                      args.jacobi_order2, args.gauss_seidel_order2, args.preconditioned_cg]):
        print("⚠️  Nenhum método especificado. Usando --all por padrão.")
        args.all = True
    
    # Validar sistemas não lineares
    if args.nonlinear:
        parser.error("Módulo nonlinear_solver não encontrado. Verifique se foi instalado corretamente.")
    
    return args
