# Z-Matrix Method

This repository implements the Z-Matrix method for **power flow analysis** in electrical systems. The approach provides an alternative to traditional formulations (such as Newton-Raphson), with advantages in certain network topologies.

## Requirements

- Python 3.7+
- Jupyter Notebook or JupyterLab

## Project Structure

```
z-matrix-method/
├── src/                    # Implementation of ZMatrix class and auxiliary methods
├── *.ipynb                 # Notebooks with examples and experiments
├── requirements.txt        # Python dependencies
├── README.md              # This file
```

### `src/`
Contains the main implementation of the `MetodoMatrizZdetailed` class, which applies the Z-Matrix method to solve power flow in electrical power networks, including:
- ZIP load models (Z-Impedance, I-Current, P-Power)
- P-V curve analysis
- Block substitution and direct methods
- Support for 4-bus and 13-bus systems

### Notebooks
Jupyter notebooks that demonstrate the use of the Z-Matrix method in test cases:
- `01_basic_4bus_system.ipynb` - Basic 4-bus system
- `02_4bus_pv_curve_analysis.ipynb` - P-V curve analysis for 4-bus system
- `03_13bus_system.ipynb` - 13-bus system
- `04_13bus_pv_curve_analysis.ipynb` - P-V curve analysis for 13-bus system

## How to use

Clone the repository:

```bash
git clone https://github.com/alanjk3/z-matrix-method.git
cd z-matrix-method
```

Install requirements:

```bash
pip install -r requirements.txt
```

Run the notebooks:

```bash
jupyter notebook
```

## Implemented Examples

- ✅ 4-bus system with constant power
- ✅ 4-bus system with P-V curve analysis
- ✅ 13-bus system with constant power  
- ✅ 13-bus system with P-V curve analysis
- ✅ ZIP load models (Z-Impedance, I-Current, P-Power)
- ✅ Block substitution method for convergence
- ✅ Voltage stability analysis

## Technical Features

- **Method**: Z-Matrix with slack bus elimination
- **Load Models**: Complete support for ZIP models
- **Convergence**: Iterative method with configurable tolerance
- **Analysis**: P-V curves for stability margin assessment
- **Systems**: Tested on 4-bus and 13-bus systems


