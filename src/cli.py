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
                %(prog)s --no-plots                 # Executar sem gráficos
                %(prog)s --clear-old-data           # Limpar resultados anteriores
                %(prog)s --all --save-solutions --clear-old-data # Execução completa limpa
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Métodos disponíveis
    method_group = parser.add_argument_group('Seleção de Métodos')
    method_group.add_argument('--all', action='store_true',
                              help='Executar todos os métodos disponíveis')
    method_group.add_argument('--jacobi', action='store_true',
                              help='Executar método de Jacobi')
    method_group.add_argument('--gauss-seidel', action='store_true',
                              help='Executar método de Gauss-Seidel')
    method_group.add_argument('--conjugate-gradient', action='store_true',
                              help='Executar Gradiente Conjugado (para matrizes simétricas pos-def)')
    method_group.add_argument('--preconditioned-cg', action='store_true',
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
    nonlinear_group = parser.add_argument_group('Seleção de Métodos Não Lineares')
    nonlinear_group.add_argument('--nonlinear', action='store_true',
                                 help='Ativar solucionadores para sistemas não lineares. Use sozinho para rodar todos os métodos não lineares.')
    nonlinear_group.add_argument('--newton', action='store_true',
                                 help='Executar método de Newton (requer --nonlinear)')
    nonlinear_group.add_argument('--gradient', action='store_true',
                                 help='Executar método do Gradiente (requer --nonlinear)')
    nonlinear_group.add_argument('--iteration', action='store_true',
                                 help='Executar método de Iteração (requer --nonlinear)')
    
    # Bacias de atração
    nonlinear_group.add_argument('--basin-map', action='store_true',
                                 help='Gerar mapa de bacias de atração (requer --nonlinear)')
    nonlinear_group.add_argument('--basin-resolution', type=int, default=100,
                                 help='Resolução da grade para o mapa de bacias (padrão: 100)')

    args = parser.parse_args()
    
    # Validar dependências de argumentos
    if args.visualize_benchmark and not args.benchmark:
        parser.error("--visualize-benchmark requer --benchmark")

    # Validar dependências de argumentos não lineares
    is_nonlinear_method_selected = any([args.newton, args.gradient, args.iteration, args.basin_map])
    if is_nonlinear_method_selected and not args.nonlinear:
        parser.error("Argumentos como --newton, --gradient, --iteration e --basin-map requerem --nonlinear")
    
    # Se nenhum método foi especificado (e não é benchmark ou não linear), usar --all
    is_method_selected = any([
        args.jacobi, args.gauss_seidel, args.conjugate_gradient, args.preconditioned_cg
    ])
    if not args.nonlinear and not args.benchmark and not args.all and not is_method_selected:
        print("⚠️  Nenhum método selecionado para análise. Usando --all por padrão.")
        args.all = True
        
    return args
