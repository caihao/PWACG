# PWACG - Partial Wave Analysis Code Generator
[Readme 中文版](documentation/README_CN.md)

PWACG (Partial Wave Analysis Code Generator) is a code generation tool specifically designed for partial wave analysis. It uses advanced code generation techniques to produce highly efficient analysis code with fast computation speeds and optimized memory usage. This tool supports the use of the Newton conjugate gradient method for optimization in large datasets, significantly improving the efficiency of searching for global optimal points in partial wave analysis fitting.

## Installation and Environment Setup

### 1. Get the Installation Package
First, clone the PWACG repository from GitHub to your local machine:

```bash
git clone https://github.com/caihao/PWACG.git
```

This will create a folder named `PWACG` in the current directory containing all the necessary files.

### 2. Install Miniconda
Please download and install Miniconda based on your operating system from the [Miniconda Official Website](https://www.anaconda.com/docs/getting-started/miniconda/main).

### 3. Install JAX
JAX has been tested on NVIDIA 30 series and the latest GPUs, requiring an already installed CUDA environment and NVIDIA drivers. Please verify that GPU support is working after installing JAX.

Install JAX according to the CUDA version:

```bash
# For CUDA 12.x
pip install -U "jax[cuda12]"
```

**Verify GPU support for JAX:**

```bash
python -c "import jax; print(jax.devices())"
```

**JAX official installation guide:** [JAX Installation Guide](https://github.com/jax-ml/jax?tab=readme-ov-file#installation)

### 4. Install ROOT
Install the ROOT data analysis framework using conda:

```bash
conda config --set channel_priority strict
conda install -c conda-forge root
```

### 5. Install Python Dependencies
Install the remaining Python dependencies:

```bash
pip install -U \
    jinja2 \
    iminuit \
    pynvml \
    matplotlib \
    pandas \
    tabulate
```

## Quick Start

Before using PWACG, ensure that you have all the necessary data and configuration files ready.

### 1. Download Demo Data

Download the `data.zip` data file from the GitHub Releases:

```bash
wget https://github.com/caihao/PWACG/releases/download/v1.0.0/data.zip
```

Extract the data:

```bash
unzip data.zip
```

The extracted directory will look like this:

```bash
$ ls data
draw_data  draw_mc  mc_int  mc_truth  real_data  weight
```

### 2. Generate Analysis Scripts

Use the following command to generate the required analysis scripts:

```bash
python create_all_scripts.py
```

This command will create a series of scripts based on the provided data and configuration, which are needed for the partial wave analysis process.

### 3. Run the Fitting Demo

After generating the scripts, you can run the fitting process:

```bash
# Generate fitting scripts
$ python create_all_scripts.py

# Run the fitting
$ python run/fit_kk.py
```

This will run the `fit_kk.py` script to begin fitting your data using partial wave analysis. The process may take some time depending on the data size and configuration.

### 4. Plot the Results

After the fitting is complete, you can generate and view the fitting results in graphical form:

```bash
# Generate plotting scripts for the fitting results
$ python create_all_scripts.py

# Generate the fitting result weights
$ python run/draw_wt_kk.py

# Plot the results
$ python run/dplot_run_kk.py
```

The plotted results will be saved in the `output/pictures/partial_mods_pictures/` directory.

## Project Highlights

- **Fast Computation**: The tool utilizes code generation techniques to optimize the execution paths of algorithms, significantly improving computation speed.
- **Efficient Memory Usage**: Smart memory management ensures high memory efficiency, suitable for handling large-scale datasets.
- **Newton Conjugate Gradient Optimization**: Supports large datasets with efficient Newton conjugate gradient optimization, enhancing the search for global optimal solutions.
- **Suitable for Large-Scale Data Analysis**: Especially designed for partial wave analysis tasks that involve handling and analyzing large volumes of data.

## Documentation

For more detailed usage instructions and API documentation, please visit the [Documentation Link](Tutorial_CN.md).

## Contributing

We welcome contributions in various forms, including but not limited to new features, bug fixes, and documentation improvements. Please share your ideas with us through Pull Requests or Issues.

## License

This project is licensed under the [MIT License](LICENSE). Please refer to the accompanying LICENSE file for details.

## Contact

If you have any questions or suggestions, please contact us via the following channels:

- GitHub Issues: https://github.com/caihao/PWACG/issues
