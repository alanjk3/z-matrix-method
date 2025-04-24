# Z-Matrix Method

Este repositório implementa o método Z-Matrix para análise de **fluxo de potência** em sistemas elétricos. A abordagem fornece uma alternativa às formulações tradicionais (como Newton-Raphson), com vantagens em determinadas topologias de rede.

##  Estrutura do Projeto

```
z-matrix-method/
├── src/              # Implementação da classe ZMatrix e métodos auxiliares
├── notebook/         # Notebooks com exemplos e experimentos
├── README.md         # Este arquivo
```

### `src/`
Contém a implementação da classe principal `ZMatrix`, que aplica o método Z-Matrix para resolver o fluxo de potência em redes de energia elétrica.

### `notebook/`
Notebooks Jupyter que demonstram o uso do método Z-Matrix em casos de teste. 

## Como usar

Clone o repositório:

```bash
git clone https://github.com/alanjk3/z-matrix-method.git
cd z-matrix-method
```

Instale os requisitos (se houver):

```bash
pip install -r requirements.txt
```

##  Exemplos

Os exemplos estão nos notebooks da pasta `notebook/`. Alguns exemplos disponíveis:

- Exemplo 4 barras com potência constante
- Exemplo 4 barras com impedância constante (a desenvolver)
- Exemplo 4 barras com corrente constante (a desenvolver)
- Exemplo 4 barras com modelo ZIP (a desenvolver)


