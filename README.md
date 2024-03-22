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
To run the fitting demo, first download the data from the releases. Unzip the data into the `data` directory. If the directory doesn't exist, create it as follows:

```bash
# Create the data directory (if it doesn't exist)
$ mkdir data
# Unzip the data into the data directory
$ unzip data/data.zip -d data
```

After unzipping, the directory structure should look like this:

```bash
$ ls data          
draw_data  draw_mc  mc_int  mc_truth  real_data  weight
```

Once the scripts are generated, you can start the fitting process with the following commands:

```bash
# Generate the script for the fitting data
$ python create_all_scripts.py
# Run the fitting
$ python run/fit.py
```

This will execute the `fit.py` script and begin the partial wave analysis fitting process on your data. Depending on the volume of data and the configuration, this process may take some time.

### Plotting the Fitting Results
To plot the fitting results, generate the plotting scripts and produce the weights, then execute the plotting script as follows:

```bash
# Generate the script for plotting the fitting results
$ python create_all_scripts.py
# Generate the weights for the fitting results
$ python run/draw_wt_kk.py
# Plot the results
$ python run/dplot_run_kk.py
```

The resulting plots will be saved in the `output/pictures/partial_mods_pictures` directory.

### Calculate the Covariant Tensor of the Angular Distribution
To compute the covariant tensor of the angular distribution from the `Momentum_kk.npz` file in each directory of the input data, you can generate and run the necessary scripts as follows:

```bash
# Generate the scripts
$ python create_all_scripts.py
# Run the calculation
$ python run/RunCacheTensor.py
```

## Documentation

For more detailed usage instructions and API documentation, please visit [Documentation Link](documentation/Tutorial_EN.md).

## Contributing

Any form of contribution is welcome, including but not limited to new features, bug fixes, documentation improvements, etc. Please share your ideas with us through Pull Requests or Issues.

## License

This project is licensed under the [MIT License](LICENSE). Please refer to the included license file for more details.

## Contact

If you have any questions or suggestions, please feel free to contact us via the following channels:

- GitHub Issues: https://github.com/caihao/PWACG/issues
