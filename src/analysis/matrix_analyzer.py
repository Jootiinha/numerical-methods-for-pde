import numpy as np
from pathlib import Path
from typing import Dict
from src.linear_solver.utils.matrix_validator import MatrixValidator

def calcular_matriz_iteracao_jacobi(A: np.ndarray) -> np.ndarray:
    """Calcula a matriz de iteração do método de Jacobi."""
    n = A.shape[0]
    D = np.diag(np.diag(A))
    D_inv = np.linalg.inv(D)
    M = np.eye(n) - D_inv @ A
    return M


def calcular_matriz_iteracao_gauss_seidel(A: np.ndarray) -> np.ndarray:
    """Calcula a matriz de iteração do método de Gauss-Seidel."""
    n = A.shape[0]
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    DL = D + L
    M = np.linalg.solve(DL, U)
    return M


def calcular_matriz_iteracao_jacobi_ordem2(A: np.ndarray, omega1: float = 0.7, 
                                          omega2: float = 0.2, omega3: float = 0.1) -> np.ndarray:
    """Calcula matriz de iteração aproximada para Jacobi de ordem 2."""
    M_jacobi = calcular_matriz_iteracao_jacobi(A)
    n = A.shape[0]
    I = np.eye(n)
    M_j2 = omega1 * M_jacobi + omega2 * I + omega3 * np.linalg.matrix_power(M_jacobi, 2)
    return M_j2


def calcular_matriz_iteracao_gauss_seidel_ordem2(A: np.ndarray, omega_relax: float = 1.2,
                                                omega1: float = 0.8, omega2: float = 0.15, 
                                                omega3: float = 0.05) -> np.ndarray:
    """Calcula matriz de iteração aproximada para Gauss-Seidel de ordem 2 com SOR."""
    n = A.shape[0]
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    D_wL = D + omega_relax * L
    rhs = (1 - omega_relax) * D - omega_relax * U
    M_sor = np.linalg.solve(D_wL, rhs)
    I = np.eye(n)
    M_gs2 = omega1 * M_sor + omega2 * I + omega3 * np.linalg.matrix_power(M_sor, 2)
    return M_gs2


def calcular_normas_matriz(M: np.ndarray) -> Dict[str, float]:
    """Calcula todas as normas solicitadas para uma matriz, incluindo versões ponderadas."""
    normas = {}
    
    # Normas básicas
    try:
        normas['Euclidiana'] = np.linalg.norm(M, 2)
    except:
        normas['Euclidiana'] = np.nan
        
    normas['Soma_Max_Colunas'] = np.linalg.norm(M, 1)
    normas['Soma_Max_Linhas'] = np.linalg.norm(M, np.inf)
    normas['Frobenius'] = np.linalg.norm(M, 'fro')
    normas['Maximo'] = np.max(np.abs(M))
    
    # Raio espectral
    try:
        eigenvals = np.linalg.eigvals(M)
        normas['Raio_Espectral'] = np.max(np.abs(eigenvals))
    except:
        normas['Raio_Espectral'] = np.nan
    
    # Versões ponderadas (usando pesos baseados na posição)
    n = M.shape[0]
    
    # Pesos lineares crescentes
    pesos_linha = np.arange(1, n+1, dtype=float)
    pesos_coluna = np.arange(1, n+1, dtype=float)
    
    # Matriz ponderada por linha
    M_pond_linha = M * pesos_linha.reshape(-1, 1)
    normas['Euclidiana_Pond_Linha'] = np.linalg.norm(M_pond_linha, 2) if M_pond_linha.size > 0 else np.nan
    normas['Frobenius_Pond_Linha'] = np.linalg.norm(M_pond_linha, 'fro')
    
    # Matriz ponderada por coluna  
    M_pond_coluna = M * pesos_coluna.reshape(1, -1)
    normas['Euclidiana_Pond_Coluna'] = np.linalg.norm(M_pond_coluna, 2) if M_pond_coluna.size > 0 else np.nan
    normas['Frobenius_Pond_Coluna'] = np.linalg.norm(M_pond_coluna, 'fro')
    
    # Matriz duplamente ponderada
    Pesos = np.outer(pesos_linha, pesos_coluna)
    M_pond_dupla = M * Pesos
    normas['Euclidiana_Pond_Dupla'] = np.linalg.norm(M_pond_dupla, 2) if M_pond_dupla.size > 0 else np.nan
    normas['Frobenius_Pond_Dupla'] = np.linalg.norm(M_pond_dupla, 'fro')
    
    return normas


def calcular_numero_condicao(A: np.ndarray) -> Dict[str, float]:
    """Calcula número de condição da matriz A para diferentes normas."""
    cond = {}
    
    try:
        cond['Euclidiana'] = np.linalg.cond(A, 2)
    except:
        cond['Euclidiana'] = np.nan
        
    cond['Soma_Max_Colunas'] = np.linalg.cond(A, 1)
    cond['Soma_Max_Linhas'] = np.linalg.cond(A, np.inf)
    
    # Para Frobenius, calcular manualmente: ||A|| * ||A^(-1)||
    try:
        A_inv = np.linalg.inv(A)
        cond['Frobenius'] = np.linalg.norm(A, 'fro') * np.linalg.norm(A_inv, 'fro')
    except:
        cond['Frobenius'] = np.nan
    
    # Para norma máximo
    try:
        A_inv = np.linalg.inv(A)
        cond['Maximo'] = np.max(np.abs(A)) * np.max(np.abs(A_inv))
    except:
        cond['Maximo'] = np.nan
    
    return cond


def analisar_condicionamento_sistema(A: np.ndarray, matrix_name: str):
    """Análise completa de condicionamento da matriz A e das matrizes de iteração."""
    
    print(f"\n📊 ANÁLISE DE CONDICIONAMENTO: {matrix_name}")
    print("-" * 60)
    
    # Calcular matrizes de iteração
    print("🔧 Calculando matrizes de iteração...")
    matrizes_iteracao = {
        'M1 (Jacobi)': calcular_matriz_iteracao_jacobi(A),
        'M2 (Gauss-Seidel)': calcular_matriz_iteracao_gauss_seidel(A),
        'M3 (Jacobi Ordem 2)': calcular_matriz_iteracao_jacobi_ordem2(A),
        'M4 (Gauss-Seidel Ordem 2)': calcular_matriz_iteracao_gauss_seidel_ordem2(A)
    }
    
    # Condicionamento da matriz A
    print("📊 Calculando condicionamento da matriz A...")
    cond_A = calcular_numero_condicao(A)
    
    # Normas das matrizes de iteração (incluindo versões ponderadas)
    print("📐 Calculando normas das matrizes de iteração...")
    normas_resultados = {}
    for nome, M in matrizes_iteracao.items():
        normas_resultados[nome] = calcular_normas_matriz(M)
    
    # Mostrar resultados resumidos no console
    print("\n📈 Condicionamento da Matriz A:")
    for norma, valor in cond_A.items():
        print(f"   {norma:20}: {valor:.2e}")
    
    print("\n🔄 Raio Espectral das Matrizes de Iteração:")
    for nome, normas in normas_resultados.items():
        rho = normas['Raio_Espectral']
        status = "✅" if rho < 1 else "❌"
        print(f"   {nome:25}: ρ = {rho:.6f} {status}")
    
    # Salvar resultados detalhados
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    nome_sistema = matrix_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    
    # 1. Relatório principal comparativo
    tabela_path = results_dir / f"condicionamento_comparativo_{nome_sistema}.txt"
    
    with open(tabela_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write(f"ANÁLISE COMPARATIVA DE CONDICIONAMENTO - {matrix_name}\n")
        f.write("=" * 100 + "\n\n")
        
        # Condicionamento da matriz A
        f.write("CONDICIONAMENTO DA MATRIZ A:\n")
        f.write("-" * 40 + "\n")
        for norma, valor in cond_A.items():
            f.write(f"{norma:25}: {valor:12.6e}\n")
        
        f.write(f"\n\nTABELA COMPARATIVA DAS MATRIZES DE ITERAÇÃO:\n")
        f.write("-" * 80 + "\n")
        
        # Cabeçalho da tabela (normas básicas)
        f.write(f"{'Método':<25} {'Raio Espect.':<12} {'Euclidiana':<12} {'Max Col':<10} {'Max Lin':<10} {'Frobenius':<12} {'Máximo':<10}\n")
        f.write("-" * 95 + "\n")
        
        # Dados da tabela (normas básicas)
        for nome, normas in normas_resultados.items():
            f.write(f"{nome:<25} {normas['Raio_Espectral']:<12.6f} {normas['Euclidiana']:<12.6f} "
                   f"{normas['Soma_Max_Colunas']:<10.6f} {normas['Soma_Max_Linhas']:<10.6f} "
                   f"{normas['Frobenius']:<12.6f} {normas['Maximo']:<10.6f}\n")
        
        f.write(f"\n\nRESUMO DE CONVERGÊNCIA:\n")
        f.write("-" * 25 + "\n")
        for nome, normas in normas_resultados.items():
            rho = normas['Raio_Espectral']
            status = "CONVERGE" if rho < 1 else "NÃO CONVERGE"
            f.write(f"{nome:25}: ρ = {rho:.6f} - {status}\n")
        
        f.write(f"\n\nINTERPRETAÇÃO:\n")
        f.write("- Raio Espectral < 1: Método garante convergência\n")
        f.write("- Menor raio espectral = convergência mais rápida\n")
        f.write("- Número de condição da matriz A indica estabilidade numérica\n")
    
    # 2. Relatório detalhado com normas ponderadas
    relatorio_detalhado_path = results_dir / f"analise_condicionamento_detalhada_{nome_sistema}.txt"
    
    with open(relatorio_detalhado_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"RELATÓRIO DETALHADO - ANÁLISE DE CONDICIONAMENTO\n")
        f.write(f"Sistema: {matrix_name}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("CONDICIONAMENTO DA MATRIZ ORIGINAL A:\n")
        f.write("-" * 40 + "\n")
        for norma, valor in cond_A.items():
            f.write(f"{norma:25}: {valor:.6e}\n")
        
        f.write(f"\n\nNORMAS DAS MATRIZES DE ITERAÇÃO (COMPLETAS):\n")
        f.write("-" * 50 + "\n\n")
        
        for nome, dados in normas_resultados.items():
            f.write(f"{nome}:\n")
            f.write("-" * len(nome) + "\n")
            for norma, valor in dados.items():
                f.write(f"  {norma:25}: {valor:.6e}\n")
            f.write("\n")
        
        f.write("INTERPRETAÇÃO DETALHADA:\n")
        f.write("-" * 25 + "\n")
        f.write("- Raio Espectral < 1: Método converge\n")
        f.write("- Número de Condição alto: Sistema mal condicionado\n") 
        f.write("- Normas das matrizes de iteração indicam velocidade de convergência\n")
        f.write("- Normas ponderadas mostram sensibilidade a diferentes regiões da matriz\n")
        f.write("\nMenores normas das matrizes de iteração = convergência mais rápida\n")
    
    # 3. Arquivo CSV-like para normas das matrizes de iteração
    normas_tabular_path = results_dir / f"normas_matrizes_iteracao_{nome_sistema}.txt"
    
    with open(normas_tabular_path, 'w', encoding='utf-8') as f:
        f.write("NORMAS DAS MATRIZES DE ITERAÇÃO - FORMATO TABULAR\n")
        f.write("=" * 80 + "\n\n")
        
        # Cabeçalho
        if normas_resultados:
            norma_names = list(next(iter(normas_resultados.values())).keys())
            f.write(f"{'Matriz':<25}")
            for norma in norma_names:
                f.write(f"{norma:<18}")
            f.write("\n" + "-" * (25 + 18 * len(norma_names)) + "\n")
            
            # Dados
            for nome, dados in normas_resultados.items():
                f.write(f"{nome:<25}")
                for norma in norma_names:
                    f.write(f"{dados[norma]:<18.6e}")
                f.write("\n")
    
    # 4. Arquivo separado só para condicionamento da matriz A
    cond_matriz_path = results_dir / f"condicionamento_matriz_A_{nome_sistema}.txt"
    
    with open(cond_matriz_path, 'w', encoding='utf-8') as f:
        f.write(f"CONDICIONAMENTO DA MATRIZ A - {matrix_name}\n")
        f.write("=" * 40 + "\n\n")
        for norma, valor in cond_A.items():
            f.write(f"{norma:<25}: {valor:.6e}\n")
        
        f.write(f"\nINTERPRETAÇÃO:\n")
        f.write("-" * 15 + "\n")
        euclidiana = cond_A.get('Euclidiana', float('inf'))
        if euclidiana < 10:
            f.write("Sistema MUITO BEM condicionado (κ < 10)\n")
        elif euclidiana < 100:
            f.write("Sistema BEM condicionado (10 ≤ κ < 100)\n")
        elif euclidiana < 1000:
            f.write("Sistema MODERADAMENTE condicionado (100 ≤ κ < 1000)\n")
        else:
            f.write("Sistema MAL condicionado (κ ≥ 1000)\n")
    
    print(f"\n💾 ANÁLISES SALVAS:")
    print(f"   📋 Relatório principal: {tabela_path}")
    print(f"   📊 Análise detalhada: {relatorio_detalhado_path}")
    print(f"   📄 Normas tabulares: {normas_tabular_path}")
    print(f"   📐 Condicionamento A: {cond_matriz_path}")
    
    return normas_resultados, cond_A

def analyze_matrix_properties(A, matrix_name):
    """Analisa e exibe propriedades da matriz."""
    
    print(f"\n📊 ANÁLISE DA MATRIZ: {matrix_name}")
    print("-" * 50)
    
    analysis = MatrixValidator.analyze_matrix(A)
    
    print(f"Dimensões: {analysis['shape']}")
    print(f"Determinante: {analysis['determinant']:.6e}")
    print(f"Número de condição: {analysis['condition_number']:.2e}")
    print(f"Posto: {analysis['rank']}")
    print(f"Raio espectral: {analysis['spectral_radius']:.6f}")
    
    print("\nPropriedades:")
    print(f"  ✅ Simétrica: {'Sim' if analysis['is_symmetric'] else 'Não'}")
    print(f"  ✅ Positiva definida: {'Sim' if analysis['is_positive_definite'] else 'Não'}")  
    print(f"  ✅ Diagonalmente dominante: {'Sim' if analysis['is_diagonally_dominant'] else 'Não'}")
    print(f"  ✅ Estritamente diag. dominante: {'Sim' if analysis['is_strictly_diagonally_dominant'] else 'Não'}")
    
    return analysis
