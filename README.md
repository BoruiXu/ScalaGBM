# ScalaGBM

## Overview
ScalaGBM is an efficient GPU-based GBDT system, which can handle high-dimensional and large-scale dataset and train fast. It is optimized based on ThunderGBM.

## Prerequisites
* cmake 2.8 or above
* gcc 11.x for Linux
* CUDA 11.0+

## Introduction

### Download
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
