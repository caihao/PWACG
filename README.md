# PWACG - Partial Wave Analysis Code Generator
[Readme 中文版](documentation/README_CN.md)

PWACG (Partial Wave Analysis Code Generator) is a code generation tool designed specifically for partial wave analysis. Utilizing advanced code generation techniques, it can produce analysis code with extremely fast computation speed and high memory utilization efficiency. This tool supports using the Newton conjugate gradient method for optimization in large-scale data, significantly improving the efficiency of searching for the global optimal solution in partial wave analysis fitting.

## Features

- **Blazing Fast Computation**: Significantly boosts computation speed by optimizing execution paths through code generation techniques.
- **Efficient Memory Utilization**: Intelligently manages memory resources to ensure efficient utilization, suitable for processing large-scale datasets.
- **Newton Conjugate Gradient Optimization**: Supports using the efficient Newton conjugate gradient method for optimization in large-scale data, improving the efficiency of searching for the global optimal solution.
- **Suitable for Large-Scale Data Analysis**: Particularly suitable for partial wave analysis tasks that require processing and analyzing large amounts of data.

## Installation

### Obtaining the Installation Package

First, you need to clone the PWACG repository from GitHub to your local machine:

```bash
git clone git@github.com:caihao/PWACG.git
```

This will create a folder named `PWACG` in the current directory, containing all the necessary files.

### Installing Dependencies with pip

Before installing PWACG, please ensure that you have installed JAX and all its necessary dependencies according to the requirements of [JAX](https://github.com/google/jax). Use the following command to install the minimum set of dependencies:

```bash
pip install -r requirements-min.txt
```

This will read the necessary dependencies from the `requirements-min.txt` file and install them via pip.

Make sure your Python environment meets the installation requirements of JAX, especially compatibility with CUDA and GPU, to fully leverage the performance advantages of PWACG.

## Quick Start

Before starting to use PWACG, please ensure that you have prepared all the necessary data and configuration files.

### Generate Analysis Scripts

Use the following command to generate the required analysis scripts:

```bash
python create_all_scripts.py
```

This command will create a series of scripts based on the data and configuration information you provided, which will be used in the subsequent partial wave analysis process.

### Run the Fitting Demo

After the script generation is complete, you can use the following command to start the fitting process:

```bash
python run/fit.py
```

This will run the `fit.py` script and start the partial wave analysis fitting on your data. Depending on the size of the data and the configuration, this process may take some time.

## Documentation

For more detailed usage instructions and API documentation, please visit [Documentation Link](documentation/Tutorial_EN.md).

## Contributing

Any form of contribution is welcome, including but not limited to new features, bug fixes, documentation improvements, etc. Please share your ideas with us through Pull Requests or Issues.

## License

This project is licensed under the [MIT License](LICENSE). Please refer to the included license file for more details.

## Contact

If you have any questions or suggestions, please feel free to contact us via the following channels:

- GitHub Issues: https://github.com/caihao/PWACG/issues
