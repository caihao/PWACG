# PWACG - 分波分析代码生成器

PWACG（Partial Wave Analysis Code Generator）是一款专为分波分析设计的代码生成工具，它利用先进的代码生成技术，能够产生计算速度极快且显存利用率高的分析代码。本工具支持大数据量下使用牛顿共轭梯度法进行优化，大幅提高了寻找分波分析拟合全局最优点的搜索效率。

## 特点

- **极速计算**：通过代码生成技术，优化算法执行路径，显著提升计算速度。
- **高效显存利用**：智能管理显存资源，确保高效利用，适合处理大规模数据集。
- **牛顿共轭梯度法优化**：支持大数据量下使用高效的牛顿共轭梯度法进行优化，提高搜索全局最优解的效率。
- **适用于大规模数据分析**：特别适合需要处理和分析大量数据的分波分析任务。

## 安装

### 获取安装包

首先，您需要从GitHub克隆PWACG的仓库到您的本地计算机：

```bash
git clone git@github.com:caihao/PWACG.git
```

这将在当前目录下创建一个名为 `PWACG` 的文件夹，其中包含所有必要的文件。

### 使用 pip 安装依赖

在安装PWACG之前，请确保已按照 [JAX](https://github.com/google/jax) 的要求安装了JAX及其所有必要的依赖。使用以下命令安装最小依赖集：
```bash
pip install -r requirements-min.txt
```

这将从 `requirements-min.txt` 文件中读取必要的依赖，并通过pip进行安装。

确保您的Python环境满足JAX的安装要求，特别是与CUDA和GPU的兼容性，以充分利用PWACG的性能优势。

## 快速开始

在开始使用PWACG之前，请确保您已经准备好了所有必要的数据和配置文件。

### 生成分析脚本

使用以下命令生成所需的分析脚本：

```bash
python create_all_scripts.py
```

这个命令会根据您提供的数据和配置信息，创建一系列脚本，用于后续的分波分析过程。

### 运行拟合 Demo
从releases中下载data数据。将zip数据解压到`data`目录中，如果没有请新建，操作如下：
```bash
# 创建data目录（如果不存在）
$ mkdir data
# 解压数据到data目录
$ unzip data/data.zip -d data
```
解压后的文件目录如下：
```bash
$ ls data          
draw_data  draw_mc  mc_int  mc_truth  real_data  weight
```
完成脚本生成后，您可以使用以下命令启动拟合过程：

```bash
# 产生符合数据的脚本
$ python create_all_scripts.py
# 运行拟合
$ python run/fit.py
```
这将运行 `fit.py` 脚本，开始对您的数据进行分波分析拟合。根据数据量和配置的不同，这个过程可能需要一些时间。

拟合结果画图
```bash
# 产生拟合结果的画图脚本
$ python create_all_scripts.py
# 产生拟合结果的weight
$ python run/draw_wt_kk.py
# 画图
$ python run/dplot_run_kk.py
```
画图的结果目录`output/pictures/partial_mods_pictures`

### 计算公式中的角分布协变张量部分
计算输入data中每个目录中的`Momentum_kk.npz`，并计算出张量到同一个目录中。
```
# 产生这部分代码
$ python create_all_scripts.py
# 运行
$ python run/RunCacheTensor.py
```

## 文档

更详细的使用说明和API文档，请访问 [文档链接](Tutorial_CN.md).

## 贡献

欢迎任何形式的贡献，包括但不限于新功能、代码修复、文档改进等。请通过Pull Requests或Issues与我们分享您的想法。

## 许可证

此项目采用 [MIT 许可证](LICENSE)。详细信息请查阅随附的许可证文件。

## 联系方式

如有任何问题或建议，请通过以下方式联系我们：

- GitHub Issues：https://github.com/caihao/PWACG/issues