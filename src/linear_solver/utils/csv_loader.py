"""
Utilitário para carregar matrizes e vetores de arquivos CSV.
"""

import csv
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np


class CSVMatrixLoader:
    """
    Carregador de matrizes e vetores a partir de arquivos CSV.

    Suporta diferentes formatos de entrada:
    1. Matriz aumentada [A|b] em um único arquivo
    2. Matriz A e vetor b em arquivos separados
    3. Diferentes delimitadores e formatos numéricos
    """

    @staticmethod
    def load_augmented_matrix(
        filepath: Union[str, Path], delimiter: str = ",", skip_header: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carrega sistema linear de matriz aumentada [A|b] de um arquivo CSV.

        Args:
            filepath: Caminho para o arquivo CSV
            delimiter: Delimitador usado no CSV (padrão: ',')
            skip_header: Se deve pular a primeira linha (cabeçalho)

        Returns:
            Tupla (A, b) onde A é a matriz de coeficientes e b o vetor de termos independentes

        Raises:
            FileNotFoundError: Se o arquivo não existir
            ValueError: Se o formato do arquivo for inválido
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

        try:
            # Carregar dados do CSV
            with open(filepath, "r", encoding="utf-8") as file:
                reader = csv.reader(file, delimiter=delimiter)

                # Pular cabeçalho se necessário
                if skip_header:
                    next(reader, None)

                # Ler todas as linhas
                rows = []
                for row_num, row in enumerate(reader, start=1):
                    # Filtrar células vazias e converter para float
                    numeric_row = []
                    for col_num, cell in enumerate(row):
                        cell = cell.strip()
                        if cell:  # Ignorar células vazias
                            try:
                                numeric_row.append(float(cell))
                            except ValueError:
                                raise ValueError(
                                    f"Valor não numérico encontrado na linha {row_num}, coluna {col_num + 1}: '{cell}'"
                                )

                    if numeric_row:  # Adicionar apenas linhas não vazias
                        rows.append(numeric_row)

            if not rows:
                raise ValueError("Arquivo CSV está vazio ou não contém dados válidos")

            # Converter para array numpy
            data = np.array(rows)

            # Verificar se todas as linhas têm o mesmo número de colunas
            if data.ndim != 2:
                raise ValueError("Dados devem formar uma matriz 2D")

            n_rows, n_cols = data.shape

            if n_cols < 2:
                raise ValueError(
                    "Matriz deve ter pelo menos 2 colunas (coeficientes + termo independente)"
                )

            # Separar matriz A e vetor b
            A = data[:, :-1]  # Todas as colunas exceto a última
            b = data[:, -1]  # Última coluna

            # Verificar se A é quadrada
            if A.shape[0] != A.shape[1]:
                raise ValueError(
                    f"Matriz de coeficientes deve ser quadrada. "
                    f"Encontrada matriz {A.shape[0]}x{A.shape[1]}"
                )

            return A, b

        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            else:
                raise ValueError(f"Erro ao processar arquivo CSV: {str(e)}")

    @staticmethod
    def load_separate_files(
        matrix_filepath: Union[str, Path],
        vector_filepath: Union[str, Path],
        delimiter: str = ",",
        skip_header: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carrega matriz A e vetor b de arquivos separados.

        Args:
            matrix_filepath: Caminho para arquivo da matriz A
            vector_filepath: Caminho para arquivo do vetor b
            delimiter: Delimitador usado nos CSVs
            skip_header: Se deve pular cabeçalhos

        Returns:
            Tupla (A, b)
        """
        # Carregar matriz A
        A = CSVMatrixLoader._load_matrix_from_csv(
            matrix_filepath, delimiter, skip_header
        )

        # Carregar vetor b
        b = CSVMatrixLoader._load_vector_from_csv(
            vector_filepath, delimiter, skip_header
        )

        # Verificar compatibilidade de dimensões
        if A.shape[0] != b.shape[0]:
            raise ValueError(
                f"Dimensões incompatíveis: matriz A ({A.shape[0]}x{A.shape[1]}) "
                f"e vetor b ({b.shape[0]})"
            )

        # Verificar se A é quadrada
        if A.shape[0] != A.shape[1]:
            raise ValueError(
                f"Matriz A deve ser quadrada. Encontrada: {A.shape[0]}x{A.shape[1]}"
            )

        return A, b

    @staticmethod
    def _load_matrix_from_csv(
        filepath: Union[str, Path], delimiter: str = ",", skip_header: bool = False
    ) -> np.ndarray:
        """Carrega uma matriz de um arquivo CSV."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Arquivo da matriz não encontrado: {filepath}")

        try:
            with open(filepath, "r", encoding="utf-8") as file:
                reader = csv.reader(file, delimiter=delimiter)

                if skip_header:
                    next(reader, None)

                rows = []
                for row_num, row in enumerate(reader, start=1):
                    numeric_row = []
                    for col_num, cell in enumerate(row):
                        cell = cell.strip()
                        if cell:
                            try:
                                numeric_row.append(float(cell))
                            except ValueError:
                                raise ValueError(
                                    f"Valor não numérico na matriz, linha {row_num}, coluna {col_num + 1}: '{cell}'"
                                )

                    if numeric_row:
                        rows.append(numeric_row)

            if not rows:
                raise ValueError("Arquivo da matriz está vazio")

            return np.array(rows)

        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            else:
                raise ValueError(f"Erro ao carregar matriz: {str(e)}")

    @staticmethod
    def _load_vector_from_csv(
        filepath: Union[str, Path], delimiter: str = ",", skip_header: bool = False
    ) -> np.ndarray:
        """Carrega um vetor de um arquivo CSV."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Arquivo do vetor não encontrado: {filepath}")

        try:
            with open(filepath, "r", encoding="utf-8") as file:
                reader = csv.reader(file, delimiter=delimiter)

                if skip_header:
                    next(reader, None)

                values = []
                for row_num, row in enumerate(reader, start=1):
                    for col_num, cell in enumerate(row):
                        cell = cell.strip()
                        if cell:
                            try:
                                values.append(float(cell))
                            except ValueError:
                                raise ValueError(
                                    f"Valor não numérico no vetor, linha {row_num}, coluna {col_num + 1}: '{cell}'"
                                )

            if not values:
                raise ValueError("Arquivo do vetor está vazio")

            return np.array(values)

        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            else:
                raise ValueError(f"Erro ao carregar vetor: {str(e)}")

    @staticmethod
    def save_augmented_matrix(
        A: np.ndarray,
        b: np.ndarray,
        filepath: Union[str, Path],
        delimiter: str = ",",
        precision: int = 6,
        add_header: bool = False,
    ) -> None:
        """
        Salva matriz aumentada [A|b] em arquivo CSV.

        Args:
            A: Matriz de coeficientes
            b: Vetor de termos independentes
            filepath: Caminho para salvar o arquivo
            delimiter: Delimitador a usar
            precision: Número de casas decimais
            add_header: Se deve adicionar cabeçalho
        """
        if A.shape[0] != b.shape[0]:
            raise ValueError("Dimensões incompatíveis entre A e b")

        # Concatenar A e b
        augmented = np.column_stack([A, b])

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter=delimiter)

            # Adicionar cabeçalho se solicitado
            if add_header:
                n = A.shape[1]
                headers = [f"a_{i+1}_{j+1}" for i in range(1) for j in range(n)]
                headers.append("b")
                writer.writerow(headers)

            # Escrever dados
            for row in augmented:
                formatted_row = [f"{val:.{precision}f}" for val in row]
                writer.writerow(formatted_row)

    @staticmethod
    def create_example_files(output_dir: Union[str, Path] = "examples") -> List[Path]:
        """
        Cria arquivos CSV de exemplo para teste.

        Args:
            output_dir: Diretório onde salvar os exemplos

        Returns:
            Lista dos caminhos dos arquivos criados
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        created_files = []

        # Exemplo 1: Sistema 3x3 bem condicionado
        A1 = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
        b1 = np.array([3, 2, 3])

        file1 = output_dir / "exemplo_3x3.csv"
        CSVMatrixLoader.save_augmented_matrix(A1, b1, file1, add_header=True)
        created_files.append(file1)

        # Exemplo 2: Sistema 4x4 com matriz tridiagonal
        A2 = np.array([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]])
        b2 = np.array([1, 0, 0, 1])

        file2 = output_dir / "exemplo_4x4_tridiagonal.csv"
        CSVMatrixLoader.save_augmented_matrix(A2, b2, file2, add_header=True)
        created_files.append(file2)

        # Exemplo 3: Matriz separada
        A3 = np.array([[10, 2, 1], [1, 5, 1], [2, 3, 10]])
        b3 = np.array([7, -8, 6])

        file3_A = output_dir / "matriz_A.csv"
        file3_b = output_dir / "vetor_b.csv"

        # Salvar matriz A
        with open(file3_A, "w", newline="") as f:
            writer = csv.writer(f)
            for row in A3:
                writer.writerow([f"{val:.6f}" for val in row])

        # Salvar vetor b
        with open(file3_b, "w", newline="") as f:
            writer = csv.writer(f)
            for val in b3:
                writer.writerow([f"{val:.6f}"])

        created_files.extend([file3_A, file3_b])

        return created_files
