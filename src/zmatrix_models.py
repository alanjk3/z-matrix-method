import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import conj

# Enhanced class with constant power, constant current, and constant impedance models
class MetodoMatrizZdetailed:
    def __init__(self, y_bus, barras, linhas, barras_tipo, potencias, v_inicial, 
                 load_models=None, tol=1e-5, max_iter=100):
        """
        Inicializa o método da Matriz Z para solução de fluxo de carga com múltiplos modelos de carga.
        
        Parâmetros:
        - load_models: dicionário que especifica o modelo de carga para cada barra
                      Formato: {barra_id: {'P': porcentagem_CP, 'I': porcentagem_CI, 'Z': porcentagem_CZ}}
                      onde CP = constant power, CI = constant current, CZ = constant impedance
                      As porcentagens devem somar 1.0 para cada barra
        """
        self.y_bus = y_bus
        self.barras = barras
        self.linhas = linhas
        self.barras_tipo = barras_tipo
        self.potencias = potencias
        self.v_inicial = v_inicial
        self.tol = tol
        self.max_iter = max_iter
        
        # Configuração dos modelos de carga (valor padrão é 100% potência constante)
        if load_models is None:
            self.load_models = {b: {'P': 1.0, 'I': 0.0, 'Z': 0.0} for b in barras}
        else:
            self.load_models = load_models
            
            # Verifica se os modelos de carga estão configurados corretamente
            for b in barras:
                if b not in self.load_models:
                    self.load_models[b] = {'P': 1.0, 'I': 0.0, 'Z': 0.0}
                else:
                    # Garante que as proporções somem 1.0
                    total = sum(self.load_models[b].values())
                    if abs(total - 1.0) > 1e-10:
                        print(f"Aviso: Proporções dos modelos de carga para barra {b} não somam 1.0. Ajustando automaticamente.")
                        factor = 1.0 / total
                        for k in self.load_models[b]:
                            self.load_models[b][k] *= factor
        
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
        
        # Calcula as admitâncias equivalentes para os modelos de impedância constante
        self.y_carga_equiv = {}
        for b in self.barras_nao_slack:
            if self.load_models[b]['Z'] > 0 and abs(self.potencias[b]) > 0:
                v_nom = abs(self.v_inicial[b])
                # Y = S*/|V|² para o componente de impedância constante
                self.y_carga_equiv[b] = conj(self.potencias[b] * self.load_models[b]['Z']) / (v_nom * v_nom)
            else:
                self.y_carga_equiv[b] = 0j
    
    def calcula_constantes_c(self):
        """Calcula as constantes C para todas as barras não-slack"""
        v_slack = self.v_inicial[self.slack_bus]
        self.c = np.zeros(len(self.barras_nao_slack), dtype=complex)
        
        # C = Z * Y_slack * V_slack
        self.c = np.dot(self.z_matriz, self.y_slack_col * v_slack).flatten()
        
        # Imprimir as constantes C
        print("\nConstantes C:")
        for i, b in enumerate(self.barras_nao_slack):
            print(f"C{b} = {self.c[i]:.6f}")
        
        return self.c
    
    def calcula_corrente_injetada(self, barra, v):
        s_nominal = self.potencias[barra]
        modelo = self.load_models[barra]
        v_mag = abs(v)
        v_nom = abs(self.v_inicial[barra])
        
        # Update power based on voltage and ZIP components
        s_p = s_nominal * modelo['P']  # Constant power
        s_i = s_nominal * modelo['I'] * (v_mag / v_nom)  # Constant current - scales with |V|
        s_z = s_nominal * modelo['Z'] * (v_mag / v_nom)**2  # Constant impedance - scales with |V|²
        
        # Total updated power
        s_updated = s_p + s_i + s_z
        
        # Calculate current using the updated power
        i_total = conj(s_updated / v)
        
        return i_total
    
    def substituicao_em_bloco(self):
        """Implementa o método da matriz Z com substituição em bloco"""
        # Prepara tensões iniciais e constantes C
        v = np.array([self.v_inicial[b] for b in self.barras], dtype=complex)
        v_nao_slack = v[self.idx_nao_slack]
        self.calcula_constantes_c()
        
        # Prepara para armazenar histórico de tensões
        v_history = []
        
        # Imprime os modelos de carga configurados
        print("\nModelos de Carga Configurados:")
        for b in self.barras_nao_slack:
            print(f"Barra {b}: {self.load_models[b]['P']*100:.1f}% Potência Constante, "
                  f"{self.load_models[b]['I']*100:.1f}% Corrente Constante, "
                  f"{self.load_models[b]['Z']*100:.1f}% Impedância Constante")
        
        # Processo iterativo
        for iter_count in range(self.max_iter):
            v_history.append(v.copy())
            
            # Imprime as tensões atuais
            print(f"\nIteração {iter_count+1} - Tensões atuais:")
            for i, b in enumerate(self.barras):
                print(f"V{b} = {v[i]:.6f} (mag={abs(v[i]):.6f}, ang={np.angle(v[i], deg=True):.6f}°)")
            
            # Calcula as correntes injetadas com os modelos ZIP
            i = np.zeros(len(self.barras_nao_slack), dtype=complex)
            print(f"\nIteração {iter_count+1} - Correntes injetadas (modelo ZIP):")
            for i_local, b in enumerate(self.barras_nao_slack):
                i[i_local] = self.calcula_corrente_injetada(b, v_nao_slack[i_local])
                print(f"I{b} = {i[i_local]:.6f} (mag={abs(i[i_local]):.6f}, ang={np.angle(i[i_local], deg=True):.6f}°)")
            
            # Calcula as novas tensões
            v_novo = np.dot(self.z_matriz, i) - self.c
            
            # Imprime as novas tensões calculadas
            print(f"\nIteração {iter_count+1} - Novas tensões calculadas:")
            for i_local, b in enumerate(self.barras_nao_slack):
                print(f"V_novo{b} = {v_novo[i_local]:.6f} (mag={abs(v_novo[i_local]):.6f}, ang={np.angle(v_novo[i_local], deg=True):.6f}°)")
            
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
        
        # Imprime os modelos de carga configurados
        print("\nModelos de Carga Configurados:")
        for b in self.barras_nao_slack:
            print(f"Barra {b}: {self.load_models[b]['P']*100:.1f}% Potência Constante, "
                  f"{self.load_models[b]['I']*100:.1f}% Corrente Constante, "
                  f"{self.load_models[b]['Z']*100:.1f}% Impedância Constante")
        
        # Inicializa o vetor de correntes com as correntes iniciais
        i = np.zeros(len(self.barras_nao_slack), dtype=complex)
        for i_local, b in enumerate(self.barras_nao_slack):
            i[i_local] = self.calcula_corrente_injetada(b, v_nao_slack[i_local])
        
        # Imprime as correntes iniciais
        print("\nCorrentes iniciais:")
        for i_local, b in enumerate(self.barras_nao_slack):
            print(f"I{b} = {i[i_local]:.6f} (mag={abs(i[i_local]):.6f}, ang={np.angle(i[i_local], deg=True):.6f}°)")
        
        # Processo iterativo
        for iter_count in range(self.max_iter):
            v_history.append(v.copy())
            v_anterior = v_nao_slack.copy()
            
            # Imprime as tensões atuais
            print(f"\nIteração {iter_count+1} - Tensões atuais:")
            for i_b, b in enumerate(self.barras):
                print(f"V{b} = {v[i_b]:.6f} (mag={abs(v[i_b]):.6f}, ang={np.angle(v[i_b], deg=True):.6f}°)")
            
            # Para cada barra, atualiza sequencialmente a tensão e a corrente
            for k in range(len(self.barras_nao_slack)):
                # Calcula nova tensão para a barra k
                v_k = np.dot(self.z_matriz[k, :], i) - self.c[k]
                v_nao_slack[k] = v_k
                
                # Atualiza imediatamente a corrente para a barra k
                b = self.barras_nao_slack[k]
                i_old = i[k]
                i[k] = self.calcula_corrente_injetada(b, v_k)
                
                # Imprime atualização de tensão e corrente para a barra k
                print(f"  Atualização da barra {b}:")
                print(f"    V{b} = {v_k:.6f} (mag={abs(v_k):.6f}, ang={np.angle(v_k, deg=True):.6f}°)")
                print(f"    I{b}: {i_old:.6f} -> {i[k]:.6f}")
            
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
    
    def calcular_injecoes(self, v):
        """Calcula as injeções de potência nas barras, considerando os modelos ZIP"""
        injecoes = {}
        
        # Para cada barra (inclusive a slack)
        for idx, b in enumerate(self.barras):
            v_b = v[idx]
            
            if b == self.slack_bus:
                # Para a barra slack, calcular potência a partir das correntes nas linhas
                s_injetada = 0j
                for linha in self.linhas:
                    if b in linha:
                        de, para = linha
                        idx_de = self.barras.index(de)
                        idx_para = self.barras.index(para)
                        
                        if de == b:  # Barra slack é "de"
                            y_linha = -self.y_bus[idx_de, idx_para]
                            i_linha = (v[idx_de] - v[idx_para]) * y_linha
                            s_linha = v[idx_de] * conj(i_linha)
                            s_injetada += s_linha
                        else:  # Barra slack é "para"
                            y_linha = -self.y_bus[idx_de, idx_para]
                            i_linha = (v[idx_para] - v[idx_de]) * y_linha
                            s_linha = v[idx_para] * conj(i_linha)
                            s_injetada += s_linha
                
                injecoes[b] = s_injetada
            
            else:  # Para barras PQ
                # Componente de potência constante
                s_p = self.potencias[b] * self.load_models[b]['P']
                
                # Componente de corrente constante
                v_mag = abs(v_b)
                s_i = self.potencias[b] * self.load_models[b]['I'] * v_mag / abs(self.v_inicial[b])
                
                # Componente de impedância constante
                s_z = self.potencias[b] * self.load_models[b]['Z'] * (v_mag * v_mag) / (abs(self.v_inicial[b]) * abs(self.v_inicial[b]))
                
                injecoes[b] = s_p + s_i + s_z
        
        return injecoes
    
    def exibir_resultados(self, v, fluxos, metodo, iteracoes):
        """Exibe os resultados da solução do fluxo de carga"""
        print(f"\n--- Resultados do método da Matriz Z - {metodo} com modelos ZIP ---")
        print(f"Convergido em {iteracoes} iterações")
        
        # Tensões nas barras
        print("\nTensões nas barras:")
        print("{:<5} {:<12} {:<12} {:<20}".format("Barra", "Magnitude", "Ângulo (°)", "Tensão (pu)"))
        for i, b in enumerate(self.barras):
            magnitude = abs(v[i])
            angulo = np.angle(v[i], deg=True)
            print("{:<5} {:<12.5f} {:<12.5f} {:<20}".format(
                b, magnitude, angulo, f"{v[i]:.5f}"))
        
        # Fluxos de potência
        print("\nFluxos de potência nas linhas:")
        print("{:<5} {:<5} {:<12} {:<12} {:<12} {:<12}".format(
            "De", "Para", "P (pu)", "Q (pu)", "P reverso", "Q reverso"))
        for de, para, s_de_para, s_para_de in fluxos:
            print("{:<5} {:<5} {:<12.5f} {:<12.5f} {:<12.5f} {:<12.5f}".format(
                de, para, s_de_para.real, s_de_para.imag, 
                s_para_de.real, s_para_de.imag))
        
        # Injeções de potência por barra
        injecoes = self.calcular_injecoes(v)
        print("\nInjeções de potência por barra (considerando modelos ZIP):")
        print("{:<5} {:<12} {:<12} {:<12}".format("Barra", "P (pu)", "Q (pu)", "S (pu)"))
        for b in self.barras:
            s = injecoes[b]
            print("{:<5} {:<12.5f} {:<12.5f} {:<12.5f}".format(
                b, s.real, s.imag, abs(s)))
    
    def plotar_convergencia(self, v_history, metodo):
        """Plota a convergência das magnitudes de tensão"""
        plt.figure(figsize=(10, 6))
        
        for i, b in enumerate(self.barras):
            if b != self.slack_bus:  # Ignora a barra slack
                magnitudes = [abs(vh[i]) for vh in v_history]
                plt.plot(magnitudes, marker='o', label=f'Barra {b}')
        
        plt.title(f'Convergência do Método da Matriz Z - {metodo} com modelos ZIP')
        plt.xlabel('Iteração')
        plt.ylabel('Magnitude da Tensão (pu)')
        plt.grid(True)
        plt.legend()  
        plt.tight_layout()
        plt.show()

def exemplo_sistema_4_barras():
    # Definir o sistema
    barras = [1, 2, 3, 4]
    
    # Barra 1 é a slack (tipo 0)
    # Barras 2, 3 e 4 são PQ (tipo 2)
    barras_tipo = {1: 0, 2: 2, 3: 2, 4: 2}  # 0: slack, 1: PV, 2: PQ
    
    # Potências complexas (S = P + jQ) em pu
    # Nota: as cargas são negativas do ponto de vista da injeção de potência
    potencias = {
        1: 0.0 - 0.7j,  # Barra slack
        2: -1.28 - 1.28j,
        3: -0.32 - 0.16j,
        4: -1.6 - 0.80j
    }
    
    # Tensões iniciais (flat start com valores de 1.0 pu, exceto a barra slack)
    v_inicial = {
        1: 1.03 + 0.0j,  # Tensão de referência na barra slack
        2: 1.0 + 0.0j,
        3: 1.0 + 0.0j,
        4: 1.0 + 0.0j
    }
    
    # Definir os modelos de carga para cada barra (ZIP)
    # Formato: {barra: {'P': porcentagem_CP, 'I': porcentagem_CI, 'Z': porcentagem_CZ}}
    # Exemplo: Barra 2 com 60% potência constante, 30% corrente constante, 10% impedância constante
    load_models = {
        1: {'P': 1.0, 'I': 0.0, 'Z': 0.0},  # Barra slack (não usado)
        2: {'P': 1.0, 'I': 0.0, 'Z': 0.0},  # Modelo ZIP
        3: {'P': 1.0, 'I': 0.0, 'Z': 0.0},  # 100% corrente constante
        4: {'P': 1.0, 'I': 0.0, 'Z': 0.0}  # 100% impedância constante
    }
    
    # Linhas (conexões entre barras)
    linhas = [(1, 2), (2, 3), (2, 4)]
    
    # Calcular os elementos da matriz Y_bus
    
    # Impedâncias das linhas
    z12 = 0.0236 + 0.0233j
    z23 = 0.045 + 0.030j
    z24 = 0.0051 + 0.0005j
    
    # Admitâncias série das linhas
    y12 = 1 / z12
    y23 = 1 / z23
    y24 = 1 / z24
    
    # Admitância shunt total por linha (j0.01 pu)
    y_shunt = 0.01j
    
    # Construir a matriz de admitância Y_bus
    Y_bus = np.zeros((4, 4), dtype=complex)
    
    # Elementos diagonais (soma das admitâncias conectadas à barra)
    Y_bus[0, 0] = y12 + y_shunt/2  # Barra 1
    Y_bus[1, 1] = y12 + y23 + y24 + y_shunt/2 + y_shunt/2 + y_shunt/2  # Barra 2
    Y_bus[2, 2] = y23 + y_shunt/2  # Barra 3
    Y_bus[3, 3] = y24 + y_shunt/2  # Barra 4
    
    # Elementos não-diagonais (negativo da admitância entre as barras)
    Y_bus[0, 1] = Y_bus[1, 0] = -y12
    Y_bus[1, 2] = Y_bus[2, 1] = -y23
    Y_bus[1, 3] = Y_bus[3, 1] = -y24
    
    print("Matriz de Admitância Y_bus:")
    for i in range(4):
        row_str = ""
        for j in range(4):
            real = Y_bus[i, j].real
            imag = Y_bus[i, j].imag
            row_str += f"{real:.6f}{'+' if imag >= 0 else ''}{imag:.6f}j  "
        print(row_str)

    # Calcular e imprimir a matriz Z_bus (inversa da Y_bus reduzida)
    # Identifica a barra slack
    slack_bus = 1
    # Índices das barras não-slack
    barras_nao_slack = [b for b in barras if b != slack_bus]
    
    # Mapear índices originais para índices na matriz
    idx_map = {b: i for i, b in enumerate(barras)}
    idx_nao_slack = [idx_map[b] for b in barras_nao_slack]
    
    # Extrair a parte da Y_bus que corresponde às barras não-slack
    Y_reduzida = Y_bus[np.ix_(idx_nao_slack, idx_nao_slack)]
    
    # Calcular a matriz Z_bus (inversa da Y_bus reduzida)
    Z_bus = np.linalg.inv(Y_reduzida)
    
    print("\nMatriz de Impedância Z_bus (inversa da Y_bus reduzida):")
    print("Barras não-slack:", barras_nao_slack)
    for i, barra_i in enumerate(barras_nao_slack):
        row_str = f"Barra {barra_i}: "
        for j, barra_j in enumerate(barras_nao_slack):
            real = Z_bus[i, j].real
            imag = Z_bus[i, j].imag
            sign = "+" if imag >= 0 else ""
            row_str += f"Z{barra_i}{barra_j}={real:.6f}{sign}{imag:.6f}j  "
        print(row_str)
    
    # Tolerância conforme informado
    tolerancia = 0.0010
    
    # Criar instância do solver com os modelos de carga
    solver = MetodoMatrizZdetailed(Y_bus, barras, linhas, barras_tipo, potencias, 
                               v_inicial, load_models=load_models, tol=tolerancia)
    
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
    
# Comparar resultados para diferentes combinações de modelos de carga
    print("\n--- Análise de Sensibilidade dos Modelos de Carga ---")
    
    # Definir diferentes cenários de modelos de carga
    load_scenarios = [
        {"name": "100% Potência Constante", "models": {
            1: {'P': 1.0, 'I': 0.0, 'Z': 0.0},
            2: {'P': 1.0, 'I': 0.0, 'Z': 0.0},
            3: {'P': 1.0, 'I': 0.0, 'Z': 0.0},
            4: {'P': 1.0, 'I': 0.0, 'Z': 0.0}
        }},
        {"name": "100% Corrente Constante", "models": {
            1: {'P': 0.0, 'I': 1.0, 'Z': 0.0},
            2: {'P': 0.0, 'I': 1.0, 'Z': 0.0},
            3: {'P': 0.0, 'I': 1.0, 'Z': 0.0},
            4: {'P': 0.0, 'I': 1.0, 'Z': 0.0}
        }},
        {"name": "100% Impedância Constante", "models": {
            1: {'P': 0.0, 'I': 0.0, 'Z': 1.0},
            2: {'P': 0.0, 'I': 0.0, 'Z': 1.0},
            3: {'P': 0.0, 'I': 0.0, 'Z': 1.0},
            4: {'P': 0.0, 'I': 0.0, 'Z': 1.0}
        }},
        {"name": "Modelo Misto (ZIP)", "models": {
            1: {'P': 0.4, 'I': 0.3, 'Z': 0.3},
            2: {'P': 0.4, 'I': 0.3, 'Z': 0.3},
            3: {'P': 0.4, 'I': 0.3, 'Z': 0.3},
            4: {'P': 0.4, 'I': 0.3, 'Z': 0.3}
        }}
    ]
    
    # Executar análise para cada cenário
    resultados_cenarios = []
    
    for scenario in load_scenarios:
        print(f"\nExecutando cenário: {scenario['name']}")
        solver_cenario = MetodoMatrizZdetailed(Y_bus, barras, linhas, barras_tipo, 
                                          potencias, v_inicial, 
                                          load_models=scenario['models'], 
                                          tol=tolerancia)
        
        # Usar método de substituição em bloco para todos os cenários
        v_final, _, iter_count = solver_cenario.substituicao_em_bloco()
        
        # Armazenar resultados
        v_magnitudes = [abs(v) for v in v_final]
        v_angulos = [np.angle(v, deg=True) for v in v_final]
        
        resultados_cenarios.append({
            "nome": scenario['name'],
            "iteracoes": iter_count,
            "magnitudes": v_magnitudes,
            "angulos": v_angulos
        })
    
    # Comparar resultados entre cenários
    print("\n--- Comparação de Resultados entre Cenários de Carga ---")
    print("{:<30} {:<10} {:<40} {:<40}".format(
        "Cenário", "Iterações", "Magnitudes de Tensão (pu)", "Ângulos de Fase (°)"))
    
    for resultado in resultados_cenarios:
        mags_str = ", ".join([f"{m:.4f}" for m in resultado["magnitudes"]])
        angs_str = ", ".join([f"{a:.2f}" for a in resultado["angulos"]])
        print("{:<30} {:<10} {:<40} {:<40}".format(
            resultado["nome"], resultado["iteracoes"], mags_str, angs_str))
    
    # Plotar comparação gráfica das magnitudes de tensão
    plt.figure(figsize=(12, 6))
    bar_width = 0.2
    x = np.arange(len(barras))
    
    for i, resultado in enumerate(resultados_cenarios):
        plt.bar(x + i*bar_width, resultado["magnitudes"], 
                width=bar_width, label=resultado["nome"])
    
    plt.xlabel('Barras')
    plt.ylabel('Magnitude da Tensão (pu)')
    plt.title('Comparação das Magnitudes de Tensão para Diferentes Modelos de Carga')
    plt.xticks(x + bar_width*1.5, [f'Barra {b}' for b in barras])
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(loc='lower right') 
    plt.tight_layout()
    plt.show()

def run_example_with_custom_load_models():
    """Executa o exemplo com modelos de carga personalizados definidos pelo usuário"""
    # Definir o sistema
    barras = [1, 2, 3, 4]
    barras_tipo = {1: 0, 2: 2, 3: 2, 4: 2}  # 0: slack, 1: PV, 2: PQ
    
    # Potências complexas (S = P + jQ) em pu
    potencias = {
        1: 0.0 - 0.7j,  # Barra slack
        2: -1.28 - 1.28j,
        3: -0.32 - 0.16j,
        4: -1.6 - 0.80j
    }
    
    # Tensões iniciais
    v_inicial = {
        1: 1.03 + 0.0j,  # Tensão de referência na barra slack
        2: 1.0 + 0.0j,
        3: 1.0 + 0.0j,
        4: 1.0 + 0.0j
    }
    
    # Linhas (conexões entre barras)
    linhas = [(1, 2), (2, 3), (2, 4)]
    
    # Impedâncias das linhas
    z12 = 0.0236 + 0.0233j
    z23 = 0.045 + 0.030j
    z24 = 0.0051 + 0.0005j
    
    # Admitâncias série das linhas
    y12 = 1 / z12
    y23 = 1 / z23
    y24 = 1 / z24
    
    # Admitância shunt total por linha (j0.01 pu)
    y_shunt = 0.01j
    
    # Construir a matriz de admitância Y_bus
    Y_bus = np.zeros((4, 4), dtype=complex)
    
    # Elementos diagonais
    Y_bus[0, 0] = y12 + y_shunt/2  # Barra 1
    Y_bus[1, 1] = y12 + y23 + y24 + y_shunt/2 + y_shunt/2 + y_shunt/2  # Barra 2
    Y_bus[2, 2] = y23 + y_shunt/2  # Barra 3
    Y_bus[3, 3] = y24 + y_shunt/2  # Barra 4
    
    # Elementos não-diagonais
    Y_bus[0, 1] = Y_bus[1, 0] = -y12
    Y_bus[1, 2] = Y_bus[2, 1] = -y23
    Y_bus[1, 3] = Y_bus[3, 1] = -y24
    
    # Solicitar ao usuário os modelos de carga para cada barra
    load_models = {}
    print("\n=== Configuração dos Modelos de Carga ===")
    print("Para cada barra, defina a porcentagem de cada modelo (deve somar 100%):")
    
    for b in [2, 3, 4]:  # Solicitar apenas para barras PQ (não slack)
        print(f"\nBarra {b}:")
        while True:
            try:
                p_const = float(input(f"  % Potência Constante (P): ")) / 100.0
                i_const = float(input(f"  % Corrente Constante (I): ")) / 100.0
                z_const = float(input(f"  % Impedância Constante (Z): ")) / 100.0
                
                total = p_const + i_const + z_const
                if abs(total - 1.0) > 1e-6:
                    print(f"Erro: O total deve ser 100%. Atual: {total*100:.1f}%")
                    continue
                
                load_models[b] = {'P': p_const, 'I': i_const, 'Z': z_const}
                break
            except ValueError:
                print("Entrada inválida. Por favor, insira um número.")
    
    # Definir modelo para barra slack (não usado nos cálculos)
    load_models[1] = {'P': 1.0, 'I': 0.0, 'Z': 0.0}
    
    # Tolerância
    tolerancia = 0.0010
    
    # Criar instância do solver
    solver = MetodoMatrizZdetailed(Y_bus, barras, linhas, barras_tipo, potencias, 
                               v_inicial, load_models=load_models, tol=tolerancia)
    
    # Executar os métodos
    print("\n=== Método da Matriz Z com Substituição em Bloco ===")
    v_bloco, v_history_bloco, iteracoes_bloco = solver.substituicao_em_bloco()
    fluxos_bloco = solver.calcular_fluxos(v_bloco)
    solver.exibir_resultados(v_bloco, fluxos_bloco, "Substituição em Bloco", iteracoes_bloco)
    
    print("\n=== Método da Matriz Z com Substituição Direta ===")
    v_direta, v_history_direta, iteracoes_direta = solver.substituicao_direta()
    fluxos_direta = solver.calcular_fluxos(v_direta)
    solver.exibir_resultados(v_direta, fluxos_direta, "Substituição Direta", iteracoes_direta)
    
    # Plotar resultados
    solver.plotar_convergencia(v_history_bloco, "Substituição em Bloco")
    solver.plotar_convergencia(v_history_direta, "Substituição Direta")

if __name__ == "__main__":
    # Executar o exemplo padrão com modelos de carga predefinidos
    exemplo_sistema_4_barras()
    
    # Descomente a linha abaixo para permitir que o usuário defina os modelos de carga
    # run_example_with_custom_load_models()