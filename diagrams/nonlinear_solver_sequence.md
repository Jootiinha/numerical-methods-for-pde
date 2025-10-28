# Diagrama de Sequência Detalhado para o Método de Newton

Este diagrama de sequência ilustra a interação entre os componentes de software ao resolver um sistema de equações não lineares `F(x) = 0` usando o método de Newton. Ele mostra como a aplicação cliente coordena o solver principal, que, por sua vez, utiliza módulos para calcular a Jacobiana e resolver um sistema linear a cada iteração.

```mermaid
sequenceDiagram
    autonumber
    actor User as "Usuário"
    participant App as "Aplicação Cliente"
    participant NewtonSolver as "Solver (Newton)"
    participant JacobianModule as "Módulo Jacobiana"
    participant LinearSystemSolver as "Solver Linear"

    User->>App: Solicitar solução de F(x)=0
    App->>+NewtonSolver: solve(F, x_initial, tol, max_iter)
    
    NewtonSolver->>NewtonSolver: Inicializar k=0, x_k = x_initial
    
    loop Iteração Principal (enquanto não converge)
        NewtonSolver->>NewtonSolver: Calcular Vetor de Resíduos F(x_k)
        
        alt Critério de Parada Satisfeito (||F(x_k)|| < tol)
            note over NewtonSolver: Convergência alcançada!
            NewtonSolver-->>-App: Retorna Solução Convergida x_k
            App-->>User: Exibe Solução
        else Ou o loop continua
            NewtonSolver->>+JacobianModule: Calcular Jacobiana em x_k
            JacobianModule-->>-NewtonSolver: Retorna Matriz J(x_k)
            
            note over NewtonSolver, LinearSystemSolver: Resolve o sistema J(x_k) * Δx = -F(x_k) para encontrar o passo de Newton
            
            NewtonSolver->>+LinearSystemSolver: solve(J(x_k), -F(x_k))
            LinearSystemSolver-->>-NewtonSolver: Retorna o passo Δx
            
            NewtonSolver->>NewtonSolver: Atualiza a solução: x_{k+1} = x_k + Δx
            NewtonSolver->>NewtonSolver: k++
        end
    end
    
    alt Atingiu o Máximo de Iterações
        NewtonSolver-->>-App: Retorna Erro de Convergência
        App-->>User: Exibe "Falha na convergência"
    end
