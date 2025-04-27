import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import conj

class MetodoMatrizZ:
    def __init__(self, y_bus, barras, linhas, barras_tipo, potencias, v_inicial, tol=1e-5, max_iter=100):
        """
        Inicializa o método da Matriz Z para solução de fluxo de carga.
        
        Parâmetros:
        -----------
        y_bus : numpy.ndarray
            Matriz de admitância do sistema (complexa)
        barras : list
            Lista com os números das barras
        linhas : list
            Lista de tuplas com as barras conectadas por cada linha
        barras_tipo : dict
            Dicionário com os tipos de cada barra (0: slack, 1: PV, 2: PQ)
        potencias : dict
            Dicionário com as potências complexas especificadas para cada barra
        v_inicial : dict
            Dicionário com as tensões iniciais para cada barra
        tol : float
            Tolerância para convergência
        max_iter : int
            Número máximo de iterações
        """
        self.y_bus = y_bus
        self.barras = barras
        self.linhas = linhas
        self.barras_tipo = barras_tipo
        self.potencias = potencias
        self.v_inicial = v_inicial
        self.tol = tol
        self.max_iter = max_iter
        
        # Identifica barra de referência (slack)
        self.slack_bus = [b for b in barras if barras_tipo[b] == 0][0]
        
        # Prepara as matrizes reduzidas (excluindo a barra slack)
        self.barras_nao_slack = [b for b in barras if b != self.slack_bus]
        
        # Mapeia índices originais para índices reduzidos
        idx_map = {b: i for i, b in enumerate(barras)}
        self.idx_nao_slack = [idx_map[b] for b in self.barras_nao_slack]
        
        # Reduz a matriz de admitância
        idx = self.idx_nao_slack
        self.y_reduzida = self.y_bus[np.ix_(idx, idx)]
        
        # Cria a matriz Z invertendo a matriz Y reduzida
        self.z_matriz = np.linalg.inv(self.y_reduzida)
        
        # Obtém os índices da barra slack na matriz original
        self.idx_slack = idx_map[self.slack_bus]
        
        # Extrai as colunas de Y relacionadas à barra slack para os cálculos de C
        self.y_slack_col = self.y_bus[np.ix_(idx, [self.idx_slack])]
    
    def calcula_constantes_c(self):
        """Calcula as constantes C para todas as barras não-slack"""
        v_slack = self.v_inicial[self.slack_bus]
        self.c = np.zeros(len(self.barras_nao_slack), dtype=complex)
        
        # C = Z * Y_slack * V_slack
        self.c = np.dot(self.z_matriz, self.y_slack_col * v_slack).flatten()
        
        return self.c
    
    def substituicao_em_bloco(self):
        """Implementa o método da matriz Z com substituição em bloco"""
        # Prepara tensões iniciais e constantes C
        v = np.array([self.v_inicial[b] for b in self.barras], dtype=complex)
        v_nao_slack = v[self.idx_nao_slack]
        self.calcula_constantes_c()
        
        # Prepara para armazenar histórico de tensões
        v_history = []
        
        # Processo iterativo
        for iter_count in range(self.max_iter):
            v_history.append(v.copy())
            
            # Calcula as correntes injetadas
            i = np.zeros(len(self.barras_nao_slack), dtype=complex)
            for i_local, b in enumerate(self.barras_nao_slack):
                i[i_local] = conj(self.potencias[b] / v_nao_slack[i_local])
            
            # Calcula as novas tensões
            v_novo = np.dot(self.z_matriz, i) - self.c
            
            # Verifica convergência
            if np.all(np.abs(v_novo - v_nao_slack) < self.tol):
                # Atualiza tensões finais no vetor completo
                v[self.idx_nao_slack] = v_novo
                return v, v_history, iter_count + 1
            
            # Atualiza tensões para próxima iteração
            v_nao_slack = v_novo
            v[self.idx_nao_slack] = v_nao_slack
        
        print(f"Atenção: Atingido número máximo de iterações ({self.max_iter})")
        return v, v_history, self.max_iter
    
    def substituicao_direta(self):
        """Implementa o método da matriz Z com substituição direta"""
        # Prepara tensões iniciais e constantes C
        v = np.array([self.v_inicial[b] for b in self.barras], dtype=complex)
        v_nao_slack = v[self.idx_nao_slack]
        self.calcula_constantes_c()
        
        # Prepara para armazenar histórico de tensões
        v_history = []
        
        # Inicializa o vetor de correntes com as correntes iniciais
        i = np.zeros(len(self.barras_nao_slack), dtype=complex)
        for i_local, b in enumerate(self.barras_nao_slack):
            i[i_local] = conj(self.potencias[b] / v_nao_slack[i_local])
        
        # Processo iterativo
        for iter_count in range(self.max_iter):
            v_history.append(v.copy())
            v_anterior = v_nao_slack.copy()
            
            # Para cada barra, atualiza sequencialmente a tensão e a corrente
            for k in range(len(self.barras_nao_slack)):
                # Calcula nova tensão para a barra k
                v_k = np.dot(self.z_matriz[k, :], i) - self.c[k]
                v_nao_slack[k] = v_k
                
                # Atualiza imediatamente a corrente para a barra k
                b = self.barras_nao_slack[k]
                i[k] = conj(self.potencias[b] / v_k)
            
            # Verifica convergência
            if np.all(np.abs(v_nao_slack - v_anterior) < self.tol):
                # Atualiza tensões finais no vetor completo
                v[self.idx_nao_slack] = v_nao_slack
                return v, v_history, iter_count + 1
            
            v[self.idx_nao_slack] = v_nao_slack
        
        print(f"Atenção: Atingido número máximo de iterações ({self.max_iter})")
        return v, v_history, self.max_iter
    
    def calcular_fluxos(self, v):
        """Calcula os fluxos de potência nas linhas"""
        fluxos = []
        
        for linha in self.linhas:
            de, para = linha
            idx_de = self.barras.index(de)
            idx_para = self.barras.index(para)
            
            # Admitância da linha
            y_linha = -self.y_bus[idx_de, idx_para]
            
            # Tensões nas barras
            v_de = v[idx_de]
            v_para = v[idx_para]
            
            # Fluxo de potência
            i_linha = (v_de - v_para) * y_linha
            s_de_para = v_de * conj(i_linha)
            
            # Fluxo reverso
            i_linha_rev = (v_para - v_de) * y_linha
            s_para_de = v_para * conj(i_linha_rev)
            
            fluxos.append((de, para, s_de_para, s_para_de))
        
        return fluxos
    
    def exibir_resultados(self, v, fluxos, metodo, iteracoes):
        """Exibe os resultados da solução do fluxo de carga"""
        print(f"\n--- Resultados do método da Matriz Z - {metodo} ---")
        print(f"Convergido em {iteracoes} iterações")
        
        # Tensões nas barras
        print("\nTensões nas barras:")
        print("{:<5} {:<12} {:<12} {:<12}".format("Barra", "Magnitude", "Ângulo (°)", "Tensão (pu)"))
        for i, b in enumerate(self.barras):
            magnitude = abs(v[i])
            angulo = np.angle(v[i], deg=True)
            print("{:<5} {:<12.5f} {:<12.5f} {:<12}".format(
                b, magnitude, angulo, f"{v[i]:.5f}"))
        
        # Fluxos de potência
        print("\nFluxos de potência nas linhas:")
        print("{:<5} {:<5} {:<12} {:<12} {:<12} {:<12}".format(
            "De", "Para", "P (pu)", "Q (pu)", "P reverso", "Q reverso"))
        for de, para, s_de_para, s_para_de in fluxos:
            print("{:<5} {:<5} {:<12.5f} {:<12.5f} {:<12.5f} {:<12.5f}".format(
                de, para, s_de_para.real, s_de_para.imag, 
                s_para_de.real, s_para_de.imag))
    
    def plotar_convergencia(self, v_history, metodo):
        """Plota a convergência das magnitudes de tensão"""
        plt.figure(figsize=(10, 6))
        
        for i, b in enumerate(self.barras):
            if b != self.slack_bus:  # Ignora a barra slack
                magnitudes = [abs(vh[i]) for vh in v_history]
                plt.plot(magnitudes, marker='o', label=f'Barra {b}')
        
        plt.title(f'Convergência do Método da Matriz Z - {metodo}')
        plt.xlabel('Iteração')
        plt.ylabel('Magnitude da Tensão (pu)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

# Solução para o sistema de 4 barras fornecido
def resolver_sistema_4_barras():
    print("Resolvendo fluxo de carga para sistema de 4 barras usando o método da Matriz Z")
    
    # Definição das barras
    barras = [1, 2, 3, 4]
    barras_tipo = {1: 0, 2: 2, 3: 2, 4: 2}  # 0: slack, 1: PV, 2: PQ
    
    # Potências complexas (S = P + jQ) em pu - valores negativos indicam consumo
    potencias = {
        1: 0.0 + 0.7j,    # Barra slack (geração reativa)
        2: -1.28 - 1.28j, # Carga
        3: -0.32 - 0.16j, # Carga
        4: -1.6 - 0.8j    # Carga
    }
    
    # Tensões iniciais (a barra 1 é fixada em 1.03+j0)
    v_inicial = {
        1: 1.03 + 0.0j,   # Barra slack com tensão fixa
        2: 1.0 + 0.0j,
        3: 1.0 + 0.0j,
        4: 1.0 + 0.0j
    }
    
    # Conexões das linhas
    linhas = [(1, 2), (2, 3), (3, 4)]
    
    # Dados das impedâncias das linhas
    z_linhas = {
        (1, 2): 0.0236 + 0.0233j,
        (2, 3): 0.045 + 0.030j,
        (3, 4): 0.0051 + 0.0005j
    }
    
    # Admitância shunt total por linha (j0.01 pu)
    y_shunt = 0.0 + 0.01j
    
    # Construir a matriz de admitância Y
    # Primeiro, inicializar com zeros
    n_barras = len(barras)
    y_bus = np.zeros((n_barras, n_barras), dtype=complex)
    
    # Preencher a matriz Y com os elementos fora da diagonal principal
    for (de, para) in linhas:
        idx_de = barras.index(de)
        idx_para = barras.index(para)
        
        # Admitância série da linha
        y_serie = 1.0 / z_linhas[(de, para)]
        
        # Elementos fora da diagonal são negativos da admitância série
        y_bus[idx_de, idx_para] = -y_serie
        y_bus[idx_para, idx_de] = -y_serie
        
        # Adicionar metade da admitância shunt às barras em cada extremidade
        y_bus[idx_de, idx_de] += y_serie + y_shunt/2
        y_bus[idx_para, idx_para] += y_serie + y_shunt/2
    
    # Criar instância do solver com tolerância especificada
    solver = MetodoMatrizZ(y_bus, barras, linhas, barras_tipo, potencias, v_inicial, tol=0.001)
    
    # Resolver usando método da matriz Z com substituição em bloco
    v_bloco, v_history_bloco, iteracoes_bloco = solver.substituicao_em_bloco()
    fluxos_bloco = solver.calcular_fluxos(v_bloco)
    
    # Exibir resultados
    solver.exibir_resultados(v_bloco, fluxos_bloco, "Substituição em Bloco", iteracoes_bloco)
    solver.plotar_convergencia(v_history_bloco, "Substituição em Bloco")
    
    # Resolver usando método da matriz Z com substituição direta
    v_direta, v_history_direta, iteracoes_direta = solver.substituicao_direta()
    fluxos_direta = solver.calcular_fluxos(v_direta)
    
    # Exibir resultados
    solver.exibir_resultados(v_direta, fluxos_direta, "Substituição Direta", iteracoes_direta)
    solver.plotar_convergencia(v_history_direta, "Substituição Direta")
    
    # Comparar número de iterações
    print(f"\nComparação do número de iterações:")
    print(f"Método de Substituição em Bloco: {iteracoes_bloco} iterações")
    print(f"Método de Substituição Direta: {iteracoes_direta} iterações")
    
    # Calcular geração na barra slack
    v_slack = v_bloco[0]  # Tensão na barra slack
    s_slack = 0
    
    # Somar os fluxos de potência saindo da barra slack
    for de, para, s_de_para, _ in fluxos_bloco:
        if de == 1:  # Se a linha começa na barra slack
            s_slack += s_de_para
    
    print(f"\nGeração na barra slack (1):")
    print(f"P = {s_slack.real:.5f} pu")
    print(f"Q = {s_slack.imag:.5f} pu")
    
    # Cálculo de perdas no sistema
    perdas_p = 0
    perdas_q = 0
    
    for de, para, s_de_para, s_para_de in fluxos_bloco:
        perdas_p += s_de_para.real + s_para_de.real
        perdas_q += s_de_para.imag + s_para_de.imag
    
    print(f"\nPerdas totais no sistema:")
    print(f"Perdas ativas (P): {perdas_p:.5f} pu")
    print(f"Perdas reativas (Q): {perdas_q:.5f} pu")

if __name__ == "__main__":
    resolver_sistema_4_barras()