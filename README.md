# ScalaGBM

## Overview
ScalaGBM is an efficient GPU-based GBDT system, which can handle high-dimensional and large-scale dataset and train fast.
## Prerequisites
* cmake 2.8 or above
* gcc 11.x for Linux
* CUDA 11.7

## 介绍 (Introduction)

### 模型初始化 (Model Initialization)

ScalaGBM 提供了灵活的模型初始化方式，支持多种任务类型和参数配置。详细的初始化指南请参考 [docs/model_initialization.md](docs/model_initialization.md)。

ScalaGBM provides flexible model initialization methods, supporting various task types and parameter configurations. For detailed initialization guide, please refer to [docs/model_initialization.md](docs/model_initialization.md).

#### 快速开始 (Quick Start)

```bash
# 二分类任务 (Binary Classification)
./bin/scalagbm-train data=dataset/adult objective=binary:logistic n_trees=100 depth=6

# 多分类任务 (Multi-class Classification)  
./bin/scalagbm-train data=dataset/covtype objective=multi:softmax num_class=7 n_trees=100

# 回归任务 (Regression)
./bin/scalagbm-train data=dataset/housing objective=reg:linear n_trees=100 depth=6
```

#### 配置文件使用 (Using Configuration Files)

```bash
# 使用配置文件 (Use configuration file)
./bin/scalagbm-train examples/config_template.conf

# 命令行参数会覆盖配置文件中的设置 (Command line arguments override config file settings)
./bin/scalagbm-train examples/config_template.conf learning_rate=0.05 n_trees=200
```

### 下载 (Download)
```bash
git clone https://github.com/BoruiXu/ScalaGBM.git

```

Build on Linux.
Before building, it is necessary to set the architecture of the GPU on line 28 (-arch) in CMakeLists.txt.
For example, when using Nvidia RTX A6000, -arch=compute_86.
```bash
cd ScalaGNM
mkdir build
cd build
cmake ..
make -j
```

### Usage example
```bash
./bin/scalagbm-train data=dataset/datasetname objective=binary:logistic tree_method=hist n_trees=40 depth=6
```

### Datasets
All test datasts can be downloaded through the script in dataset floader.
``` bash
sh ./dataset/get_datasets.sh
```

### Parameter and Test
The meaning of parameters is the same as that in ThunderGBM. At present, only histogram-based training method is supported.
We provide a bash script (train\_test.sh) to train datasets mentioned in our paper.
Befor running this script, please copy this script into the _build_ floder. If you want to test the real-sim dataset. Please run:
```bash
sh train_test.sh real-sim
```
NOTED: all datasets need to be stored in the _dataset_ folder! 
