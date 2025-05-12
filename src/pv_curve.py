import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import conj
import copy
from matplotlib.ticker import FormatStrFormatter

class PVCurveAnalysis:
    def __init__(self, base_system):
        """Initialize with a base power system configuration"""
        self.barras = base_system['barras']
        self.barras_tipo = base_system['barras_tipo']
        self.potencias_base = base_system['potencias']
        self.v_inicial = base_system['v_inicial']
        self.Y_bus = base_system['Y_bus']
        self.linhas = base_system['linhas']
        self.slack_bus = next(b for b, tipo in self.barras_tipo.items() if tipo == 0)
        self.load_buses = [b for b in self.barras if b != self.slack_bus]
        
    def run_pv_analysis(self, load_models, max_iterations=1000, step_size=0.01, tol=1e-3):
        """
        Run P-V curve analysis increasing all loads until divergence
        
        Parameters:
        - load_models: ZIP load model configuration
        - max_iterations: Maximum power flow iterations per step
        - step_size: Load increase factor per step (0.01 = 1%)
        - tol: Convergence tolerance
        
        Returns:
        - results dictionary with load multipliers and voltage profiles
        """
        results = {
            'load_multipliers': [],
            'voltages': {b: [] for b in self.barras},
            'convergence_iterations': [],
            'converged': []
        }
        
        # Start with load multiplier = 1.0 (base case)
        load_multiplier = 1.0
        converged = True
        
        while converged:
            # Scale load powers by current multiplier
            potencias_scaled = {}
            for b in self.barras:
                if b == self.slack_bus:
                    potencias_scaled[b] = self.potencias_base[b]  # Slack bus power not scaled
                else:
                    potencias_scaled[b] = self.potencias_base[b] * load_multiplier
            
            # Create solver instance with current scaled loads
            solver = MetodoMatrizZdetailed(
                self.Y_bus, self.barras, self.linhas, self.barras_tipo,
                potencias_scaled, self.v_inicial, load_models=load_models,
                tol=tol, max_iter=max_iterations, verbose=False
            )
            
            # Solve power flow using block substitution method
            v_final, _, iterations = solver.substituicao_em_bloco()
            
            # Check if converged
            converged = iterations < max_iterations
            
            # Store results
            results['load_multipliers'].append(load_multiplier)
            for i, b in enumerate(self.barras):
                results['voltages'][b].append(abs(v_final[i]) if converged else None)
            results['convergence_iterations'].append(iterations)
            results['converged'].append(converged)
            
            # Stop if not converged
            if not converged:
                print(f"Load multiplier {load_multiplier:.4f} failed to converge")
                break
                
            # Increment load multiplier for next iteration
            load_multiplier += step_size
            
            # Print progress every 20 steps
            if len(results['load_multipliers']) % 20 == 0:
                print(f"Completed load multiplier: {load_multiplier:.2f}, voltages: " + 
                      ", ".join([f"V{b}={results['voltages'][b][-1]:.4f}" for b in self.load_buses]))
        
        return results

# Modified version of the MetodoMatrizZdetailed class with reduced output for P-V analysis
class MetodoMatrizZdetailed:
    def __init__(self, y_bus, barras, linhas, barras_tipo, potencias, v_inicial, 
                 load_models=None, tol=1e-5, max_iter=100, verbose=False):
        
        self.y_bus = y_bus
        self.barras = barras
        self.linhas = linhas
        self.barras_tipo = barras_tipo
        self.potencias = potencias
        self.v_inicial = v_inicial
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        
        # Load models configuration
        if load_models is None:
            self.load_models = {b: {'P': 1.0, 'I': 0.0, 'Z': 0.0} for b in barras}
        else:
            self.load_models = load_models
            
            # Ensure load models are valid
            for b in barras:
                if b not in self.load_models:
                    self.load_models[b] = {'P': 1.0, 'I': 0.0, 'Z': 0.0}
                else:
                    total = sum(self.load_models[b].values())
                    if abs(total - 1.0) > 1e-10:
                        if self.verbose:
                            print(f"Aviso: Proporções dos modelos de carga para barra {b} não somam 1.0. Ajustando automaticamente.")
                        factor = 1.0 / total
                        for k in self.load_models[b]:
                            self.load_models[b][k] *= factor
        
        # Identify slack bus
        self.slack_bus = next(b for b in barras if barras_tipo[b] == 0)
        
        # Prepare reduced matrices (excluding slack bus)
        self.barras_nao_slack = [b for b in barras if b != self.slack_bus]
        
        # Map original indices to reduced indices
        idx_map = {b: i for i, b in enumerate(barras)}
        self.idx_nao_slack = [idx_map[b] for b in self.barras_nao_slack]
        
        # Reduce admittance matrix
        idx = self.idx_nao_slack
        self.y_reduzida = self.y_bus[np.ix_(idx, idx)]
        
        # Create Z matrix by inverting reduced Y matrix
        self.z_matriz = np.linalg.inv(self.y_reduzida)
        
        # Get slack bus indices in original matrix
        self.idx_slack = idx_map[self.slack_bus]
        
        # Extract Y columns related to slack bus for C calculations
        self.y_slack_col = self.y_bus[np.ix_(idx, [self.idx_slack])]
        
        # Calculate equivalent admittances for constant impedance models
        self.y_carga_equiv = {}
        for b in self.barras_nao_slack:
            if self.load_models[b]['Z'] > 0 and abs(self.potencias[b]) > 0:
                v_nom = abs(self.v_inicial[b])
                # Y = S*/|V|² for constant impedance component
                self.y_carga_equiv[b] = conj(self.potencias[b] * self.load_models[b]['Z']) / (v_nom * v_nom)
            else:
                self.y_carga_equiv[b] = 0j
    
    def calcula_constantes_c(self):
        """Calculate C constants for all non-slack buses"""
        v_slack = self.v_inicial[self.slack_bus]
        self.c = np.zeros(len(self.barras_nao_slack), dtype=complex)
        
        # C = Z * Y_slack * V_slack
        self.c = np.dot(self.z_matriz, self.y_slack_col * v_slack).flatten()
        
        if self.verbose:
            print("\nConstantes C:")
            for i, b in enumerate(self.barras_nao_slack):
                print(f"C{b} = {self.c[i]:.6f}")
        
        return self.c
    
    def calcula_corrente_injetada(self, barra, v):
        """Calculate injected current considering ZIP load model"""
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
        """Implement Z-matrix method with block substitution"""
        # Prepare initial voltages and C constants
        v = np.array([self.v_inicial[b] for b in self.barras], dtype=complex)
        v_nao_slack = v[self.idx_nao_slack]
        self.calcula_constantes_c()
        
        # Prepare to store voltage history
        v_history = []
        
        if self.verbose:
            # Print configured load models
            print("\nModelos de Carga Configurados:")
            for b in self.barras_nao_slack:
                print(f"Barra {b}: {self.load_models[b]['P']*100:.1f}% Potência Constante, "
                      f"{self.load_models[b]['I']*100:.1f}% Corrente Constante, "
                      f"{self.load_models[b]['Z']*100:.1f}% Impedância Constante")
        
        # Iterative process
        for iter_count in range(self.max_iter):
            v_history.append(v.copy())
            
            if self.verbose:
                # Print current voltages
                print(f"\nIteração {iter_count+1} - Tensões atuais:")
                for i, b in enumerate(self.barras):
                    print(f"V{b} = {v[i]:.6f} (mag={abs(v[i]):.6f}, ang={np.angle(v[i], deg=True):.6f}°)")
            
            # Calculate injected currents with ZIP models
            i = np.zeros(len(self.barras_nao_slack), dtype=complex)
            
            if self.verbose:
                print(f"\nIteração {iter_count+1} - Correntes injetadas (modelo ZIP):")
                
            for i_local, b in enumerate(self.barras_nao_slack):
                i[i_local] = self.calcula_corrente_injetada(b, v_nao_slack[i_local])
                
                if self.verbose:
                    print(f"I{b} = {i[i_local]:.6f} (mag={abs(i[i_local]):.6f}, ang={np.angle(i[i_local], deg=True):.6f}°)")
            
            # Calculate new voltages
            v_novo = np.dot(self.z_matriz, i) - self.c
            
            if self.verbose:
                # Print new calculated voltages
                print(f"\nIteração {iter_count+1} - Novas tensões calculadas:")
                for i_local, b in enumerate(self.barras_nao_slack):
                    print(f"V_novo{b} = {v_novo[i_local]:.6f} (mag={abs(v_novo[i_local]):.6f}, ang={np.angle(v_novo[i_local], deg=True):.6f}°)")
            
            # Check convergence
            if np.all(np.abs(v_novo - v_nao_slack) < self.tol):
                # Update final voltages in the complete vector
                v[self.idx_nao_slack] = v_novo
                return v, v_history, iter_count + 1
            
            # Update voltages for next iteration
            v_nao_slack = v_novo
            v[self.idx_nao_slack] = v_nao_slack
        
        if self.verbose:
            print(f"Atenção: Atingido número máximo de iterações ({self.max_iter})")
        return v, v_history, self.max_iter


