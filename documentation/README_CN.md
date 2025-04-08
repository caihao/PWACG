# PWACG - 分波分析代码生成器

PWACG（Partial Wave Analysis Code Generator）是一款专为分波分析设计的代码生成工具，它利用先进的代码生成技术，能够产生计算速度极快且显存利用率高的分析代码。本工具支持大数据量下使用牛顿共轭梯度法进行优化，大幅提高了寻找分波分析拟合全局最优点的搜索效率。


## 安装与环境配置

### 1. 获取安装包
首先，从 GitHub 克隆 PWACG 仓库到您的本地计算机：

```bash
git clone https://github.com/caihao/PWACG.git
```

这将在当前目录下创建一个名为 `PWACG` 的文件夹，其中包含所有必要的文件。

### 2. 安装 Miniconda
请根据您的操作系统从 [Miniconda 官方网站](https://www.anaconda.com/docs/getting-started/miniconda/main) 下载并安装 Miniconda。

### 3. 安装 JAX
JAX 在英伟达 30 系和最新款 GPU 上经过测试，要求提前安装 CUDA 环境和英伟达驱动。请在安装 JAX 后验证其是否可以正常使用 GPU。

根据 CUDA 版本选择对应的安装命令：

```bash
# 针对 CUDA 12.x
pip install -U "jax[cuda12]"
```

**验证 GPU 是否支持 JAX：**

```bash
python -c "import jax; print(jax.devices())"
```

**JAX 官方安装教程：** [JAX Installation Guide](https://github.com/jax-ml/jax?tab=readme-ov-file#installation)

### 4. 安装 ROOT
通过 conda 安装 ROOT 数据分析框架：

```bash
conda config --set channel_priority strict
conda install -c conda-forge root
```

### 5. 安装 Python 依赖
安装其余的 Python 依赖包：

```bash
pip install -U \
    jinja2 \
    iminuit \
    pynvml \
    matplotlib \
    pandas \
    tabulate
```

## 快速开始

在开始使用 PWACG 之前，请确保您已经准备好了所有必要的数据和配置文件。

### 1. 下载 Demo 数据

从 GitHub 的 Releases 中下载 `data.zip` 数据文件：

```bash
wget https://github.com/caihao/PWACG/releases/download/v1.0.0/data.zip
```

解压数据文件：

```bash
unzip data.zip
```

解压后的文件目录如下：

```bash
$ ls data
draw_data  draw_mc  mc_int  mc_truth  real_data  weight
```

### 2. 生成分析脚本

使用以下命令生成所需的分析脚本：

```bash
python create_all_scripts.py
```

这个命令会根据您提供的数据和配置信息，创建一系列脚本，用于后续的分波分析过程。

### 3. 运行拟合 Demo

在生成脚本后，您可以运行拟合过程：

```bash
# 产生拟合脚本
$ python create_all_scripts.py

# 运行拟合
$ python run/fit_kk.py
```

这将运行 `fit_kk.py` 脚本，开始对您的数据进行分波分析拟合。根据数据量和配置的不同，这个过程可能需要一些时间。

### 4. 画图

在拟合完成后，您可以生成并查看拟合结果的图表：

```bash
# 产生拟合结果的画图脚本
$ python create_all_scripts.py

# 产生拟合结果的权重
$ python run/draw_wt_kk.py

# 画图
$ python run/dplot_run_kk.py
```

画图的结果将保存在 `output/pictures/partial_mods_pictures/` 目录下。


## 项目亮点

- **极速计算**：通过代码生成技术，优化算法执行路径，显著提升计算速度。
- **高效显存利用**：智能管理显存资源，确保高效利用，适合处理大规模数据集。
- **牛顿共轭梯度法优化**：支持大数据量下使用高效的牛顿共轭梯度法进行优化，提高搜索全局最优解的效率。
- **适用于大规模数据分析**：特别适合需要处理和分析大量数据的分波分析任务。

## 文档

更详细的使用说明和API文档，请访问 [文档链接](Tutorial_CN.md).

## 贡献

欢迎任何形式的贡献，包括但不限于新功能、代码修复、文档改进等。请通过Pull Requests或Issues与我们分享您的想法。

## 许可证

此项目采用 [MIT 许可证](LICENSE)。详细信息请查阅随附的许可证文件。

## 联系方式

如有任何问题或建议，请通过以下方式联系我们：

- GitHub Issues：https://github.com/caihao/PWACG/issues