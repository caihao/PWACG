# PWACG - Partial Wave Analysis Code Generator
[中文文档](documentation/README_CN.md) | [Tutorials](documentation/Tutorial_EN.md)

PWACG is an AI-powered code generation framework for high-performance partial wave analysis. Leveraging JAX and dynamic code generation, it enables GPU-accelerated fitting with automatic differentiation and optimized memory management.

## Features

- ⚡ **JIT Compilation**: Just-In-Time compilation with JAX for GPU acceleration
- 🧠 **Dynamic Code Generation**: Auto-generate optimized analysis code based on config
- 🎛️ **Multi-GPU Support**: Distributed training across multiple GPUs
- 💾 **Memory Optimization**: Smart tensor caching and memory reuse strategies
- 📊 **Visualization Tools**: Built-in likelihood scanning and result plotting

## Installation

```bash
# Clone repository
git clone https://github.com/caihao/PWACG.git
cd PWACG

# Create conda environment
conda create -n pwacg python=3.9
conda activate pwacg

# Install dependencies
conda install -c conda-forge root jax jaxlib cuda-nvcc cuda-toolkit
pip install -r requirements.txt
```

## Basic Usage

1. **Configure analysis parameters** in `config/` directory
2. **Generate analysis scripts**:
```bash
python create_all_scripts.py
```
3. **Run fitting** (example for φKK analysis):
```bash
# Single GPU
python run/fit_kk.py

# Multi-GPU
mpirun -np 4 python run/fit_kk.py
```

## Advanced Features

- 🚀 **Hybrid Optimization**: Combine BFGS and Newton-CG methods
- 📈 **Automatic Differentiation**: Gradient calculation via JAX autodiff
- 🧮 **Tensor Caching**: Reuse intermediate calculations with `CacheTensor`
- 📉 **Significance Scanning**: Built-in 3σ-6σ scanning utilities

## Documentation

- [Configuration Guide](config/parameters.json)
- [API Reference](documentation/API.md)
- [Case Studies](examples/)

## Citing PWACG

If using this software in research, please cite:
```bibtex
@software{PWACG,
  author = {Hao Cai},
  title = {Partial Wave Analysis Code Generator},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/caihao/PWACG}}
}
```

## License

[MIT License](LICENSE)
