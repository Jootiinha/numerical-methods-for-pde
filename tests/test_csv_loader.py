"""
Testes para o carregador de CSV.
"""

import pytest
import numpy as np
import tempfile
import csv
from pathlib import Path
from linear_solver.utils import CSVMatrixLoader


class TestCSVMatrixLoader:
    """Testes para carregamento de matrizes de CSV."""
    
    def test_load_augmented_matrix_basic(self):
        """Teste básico de carregamento de matriz aumentada."""
        # Criar arquivo temporário
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow([4, -1, 0, 3])
            writer.writerow([-1, 4, -1, 2])
            writer.writerow([0, -1, 4, 3])
            temp_path = f.name
        
        try:
            A, b = CSVMatrixLoader.load_augmented_matrix(temp_path)
            
            # Verificar dimensões
            assert A.shape == (3, 3)
            assert b.shape == (3,)
            
            # Verificar valores
            expected_A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
            expected_b = np.array([3, 2, 3])
            
            np.testing.assert_array_equal(A, expected_A)
            np.testing.assert_array_equal(b, expected_b)
            
        finally:
            Path(temp_path).unlink()  # Limpar arquivo temporário
    
    def test_load_augmented_matrix_with_header(self):
        """Teste com cabeçalho."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['a11', 'a12', 'b'])  # Cabeçalho
            writer.writerow([2, 1, 3])
            writer.writerow([1, 3, 4])
            temp_path = f.name
        
        try:
            A, b = CSVMatrixLoader.load_augmented_matrix(temp_path, skip_header=True)
            
            assert A.shape == (2, 2)
            assert b.shape == (2,)
            
        finally:
            Path(temp_path).unlink()
    
    def test_load_augmented_matrix_different_delimiter(self):
        """Teste com delimitador diferente."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("2;1;3\n")
            f.write("1;3;4\n")
            temp_path = f.name
        
        try:
            A, b = CSVMatrixLoader.load_augmented_matrix(temp_path, delimiter=';')
            
            expected_A = np.array([[2, 1], [1, 3]])
            expected_b = np.array([3, 4])
            
            np.testing.assert_array_equal(A, expected_A)
            np.testing.assert_array_equal(b, expected_b)
            
        finally:
            Path(temp_path).unlink()
    
    def test_load_augmented_matrix_file_not_found(self):
        """Teste com arquivo inexistente."""
        with pytest.raises(FileNotFoundError):
            CSVMatrixLoader.load_augmented_matrix("arquivo_inexistente.csv")
    
    def test_load_augmented_matrix_non_numeric_data(self):
        """Teste com dados não numéricos."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow([2, 1, 3])
            writer.writerow([1, 'abc', 4])  # Valor não numérico
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="não numérico"):
                CSVMatrixLoader.load_augmented_matrix(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_augmented_matrix_non_square(self):
        """Teste com matriz não quadrada."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow([2, 1, 3])  # 2 coefs + 1 termo indep = 3x2 não quadrada
            writer.writerow([1, 4, 5])
            writer.writerow([0, 2, 1])
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="quadrada"):
                CSVMatrixLoader.load_augmented_matrix(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_separate_files(self):
        """Teste carregamento de arquivos separados."""
        # Arquivo da matriz A
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f_A:
            writer = csv.writer(f_A)
            writer.writerow([4, -1])
            writer.writerow([-1, 4])
            temp_path_A = f_A.name
        
        # Arquivo do vetor b
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f_b:
            writer = csv.writer(f_b)
            writer.writerow([3])
            writer.writerow([2])
            temp_path_b = f_b.name
        
        try:
            A, b = CSVMatrixLoader.load_separate_files(temp_path_A, temp_path_b)
            
            expected_A = np.array([[4, -1], [-1, 4]])
            expected_b = np.array([3, 2])
            
            np.testing.assert_array_equal(A, expected_A)
            np.testing.assert_array_equal(b, expected_b)
            
        finally:
            Path(temp_path_A).unlink()
            Path(temp_path_b).unlink()
    
    def test_load_separate_files_incompatible_dimensions(self):
        """Teste com dimensões incompatíveis entre A e b."""
        # Matriz 2x2
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f_A:
            writer = csv.writer(f_A)
            writer.writerow([4, -1])
            writer.writerow([-1, 4])
            temp_path_A = f_A.name
        
        # Vetor com 3 elementos (incompatível)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f_b:
            writer = csv.writer(f_b)
            writer.writerow([3])
            writer.writerow([2])
            writer.writerow([1])
            temp_path_b = f_b.name
        
        try:
            with pytest.raises(ValueError, match="incompatíveis"):
                CSVMatrixLoader.load_separate_files(temp_path_A, temp_path_b)
        finally:
            Path(temp_path_A).unlink()
            Path(temp_path_b).unlink()
    
    def test_save_augmented_matrix(self):
        """Teste de salvamento de matriz aumentada."""
        A = np.array([[4, -1], [-1, 4]])
        b = np.array([3, 2])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            # Salvar
            CSVMatrixLoader.save_augmented_matrix(A, b, temp_path, precision=2)
            
            # Recarregar e verificar
            A_loaded, b_loaded = CSVMatrixLoader.load_augmented_matrix(temp_path)
            
            np.testing.assert_array_almost_equal(A, A_loaded, decimal=2)
            np.testing.assert_array_almost_equal(b, b_loaded, decimal=2)
            
        finally:
            Path(temp_path).unlink()
    
    def test_save_augmented_matrix_with_header(self):
        """Teste de salvamento com cabeçalho."""
        A = np.array([[2, 1], [1, 2]])
        b = np.array([3, 4])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            CSVMatrixLoader.save_augmented_matrix(A, b, temp_path, add_header=True)
            
            # Verificar se arquivo contém cabeçalho
            with open(temp_path, 'r') as f:
                first_line = f.readline().strip()
                assert 'a_' in first_line and 'b' in first_line
            
            # Recarregar pulando cabeçalho
            A_loaded, b_loaded = CSVMatrixLoader.load_augmented_matrix(temp_path, skip_header=True)
            
            np.testing.assert_array_equal(A, A_loaded)
            np.testing.assert_array_equal(b, b_loaded)
            
        finally:
            Path(temp_path).unlink()
    
    def test_create_example_files(self):
        """Teste da criação de arquivos de exemplo."""
        with tempfile.TemporaryDirectory() as temp_dir:
            created_files = CSVMatrixLoader.create_example_files(temp_dir)
            
            # Verificar se arquivos foram criados
            assert len(created_files) > 0
            
            for file_path in created_files:
                assert file_path.exists()
                
                # Tentar carregar cada arquivo
                if file_path.name.endswith('.csv') and 'matriz_A' not in file_path.name and 'vetor_b' not in file_path.name:
                    # Arquivo de matriz aumentada
                    A, b = CSVMatrixLoader.load_augmented_matrix(file_path, skip_header=True)
                    assert A.shape[0] == A.shape[1]  # Quadrada
                    assert A.shape[0] == b.shape[0]  # Dimensões compatíveis
