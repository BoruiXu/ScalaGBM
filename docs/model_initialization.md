# ScalaGBM 模型初始化指南 (Model Initialization Guide)

## 概述 (Overview)

ScalaGBM 是一个基于 GPU 的梯度提升决策树系统，支持高维大规模数据集的快速训练。本文档详细介绍如何进行模型的初始化。

ScalaGBM is a GPU-based Gradient Boosting Decision Tree system that supports fast training on high-dimensional and large-scale datasets. This document provides a comprehensive guide on how to perform model initialization.

## 模型初始化流程 (Model Initialization Process)

### 1. 参数初始化 (Parameter Initialization)

模型初始化的第一步是设置训练参数。ScalaGBM 使用 `GBMParam` 结构体来管理所有训练参数：

```cpp
struct GBMParam {
    int depth;                    // 树的最大深度
    int n_trees;                  // 树的数量
    float_type min_child_weight;  // 子节点最小权重
    float_type lambda;            // L2 正则化参数
    float_type gamma;             // 最小分割损失
    float column_sampling_rate;   // 列采样率
    std::string objective;        // 目标函数
    int num_class;               // 分类数量
    float learning_rate;         // 学习率
    int max_num_bin;            // 直方图最大分桶数
    float base_score;           // 基础分数
    // ... 其他参数
};
```

#### 默认参数值 (Default Parameter Values)

```cpp
model_param.depth = 6;                    // 默认树深度
model_param.n_trees = 40;                 // 默认树数量
model_param.min_child_weight = 1;         // 默认最小子节点权重
model_param.lambda = 1;                   // 默认 L2 正则化
model_param.gamma = 1;                    // 默认最小分割损失
model_param.max_num_bin = 255;            // 默认最大分桶数
model_param.learning_rate = 1;            // 默认学习率
model_param.objective = "reg:linear";     // 默认目标函数
model_param.num_class = 1;                // 默认分类数
model_param.column_sampling_rate = 1;     // 默认列采样率
model_param.tree_method = "auto";         // 默认树构建方法
```

### 2. 命令行参数解析 (Command Line Parameter Parsing)

参数可以通过命令行或配置文件进行设置：

```bash
# 基本用法
./bin/scalagbm-train data=dataset/data.txt objective=binary:logistic n_trees=100 depth=8

# 完整参数示例
./bin/scalagbm-train data=dataset/adult \
    objective=binary:logistic \
    tree_method=hist \
    n_trees=100 \
    depth=6 \
    learning_rate=0.1 \
    lambda=1.0 \
    gamma=0.0 \
    max_num_bin=255 \
    n_gpus=1 \
    verbose=1
```

#### 支持的参数 (Supported Parameters)

| 参数名 | 别名 | 类型 | 描述 |
|--------|------|------|------|
| `depth` | `max_depth` | int | 树的最大深度 |
| `n_trees` | `num_round` | int | 迭代轮数（树的数量） |
| `learning_rate` | `eta` | float | 学习率 |
| `objective` | - | string | 目标函数 |
| `tree_method` | - | string | 树构建方法 |
| `max_num_bin` | `max_bin` | int | 直方图最大分桶数 |
| `lambda` | `lambda_tgbm` | float | L2 正则化参数 |
| `gamma` | `min_split_loss` | float | 最小分割损失 |
| `n_gpus` | - | int | 使用的 GPU 数量 |
| `data` | - | string | 训练数据路径 |

### 3. Booster 初始化 (Booster Initialization)

`Booster` 类负责模型的核心初始化：

```cpp
void Booster::init(const DataSet &dataSet, GBMParam &param) {
    // 1. GPU 设备管理
    int n_available_device;
    cudaGetDeviceCount(&n_available_device);
    CHECK_GE(n_available_device, param.n_device);
    
    // 2. 初始化函数构建器
    fbuilder.reset(FunctionBuilder::create(param.tree_method));
    fbuilder->init(dataSet, param);
    
    // 3. 初始化目标函数
    obj.reset(ObjectiveFunction::create(param.objective));
    obj->configure(param, dataSet);
    
    // 4. 初始化评估指标
    metric.reset(Metric::create(obj->default_metric_name()));
    metric->configure(param, dataSet);
    
    // 5. 分配内存和数据
    int n_outputs = param.num_class * dataSet.n_instances();
    gradients = MSyncArray<GHPair>(n_devices, n_outputs);
    y = MSyncArray<float_type>(n_devices, dataSet.n_instances());
    
    // 6. 复制标签数据到 GPU
    DO_ON_MULTI_DEVICES(n_devices, [&](int device_id) {
        y[device_id].copy_from(dataSet.y.data(), dataSet.n_instances());
    });
    
    // 7. 初始化基础分数
    DO_ON_MULTI_DEVICES(n_devices, [&](int device_id){
        param.base_score = obj->init_base_score(
            y[device_id], 
            fbuilder->get_raw_y_predict()[device_id], 
            gradients[device_id]
        );
    });
}
```

## 使用示例 (Usage Examples)

### 二分类任务 (Binary Classification)

```bash
./bin/scalagbm-train \
    data=dataset/adult \
    objective=binary:logistic \
    tree_method=hist \
    n_trees=100 \
    depth=6 \
    learning_rate=0.1
```

### 多分类任务 (Multi-class Classification)

```bash
./bin/scalagbm-train \
    data=dataset/covtype \
    objective=multi:softmax \
    num_class=7 \
    tree_method=hist \
    n_trees=100 \
    depth=6 \
    learning_rate=0.1
```

### 回归任务 (Regression)

```bash
./bin/scalagbm-train \
    data=dataset/housing \
    objective=reg:linear \
    tree_method=hist \
    n_trees=100 \
    depth=6 \
    learning_rate=0.1
```

## 目标函数类型 (Objective Function Types)

| 目标函数 | 描述 | 适用场景 |
|----------|------|----------|
| `binary:logistic` | 逻辑回归 | 二分类 |
| `multi:softmax` | 多分类 | 多分类 |
| `multi:softprob` | 多分类概率 | 多分类（输出概率） |
| `reg:linear` | 线性回归 | 回归 |
| `reg:logistic` | 逻辑回归 | 回归 |

## 树构建方法 (Tree Building Methods)

- `hist`: 基于直方图的方法（推荐，速度快）
- `exact`: 精确分割查找方法（精度高但速度慢）
- `auto`: 自动选择（目前默认使用 hist）

## 最佳实践 (Best Practices)

### 1. 参数调优建议

- **学习率**: 通常设置为 0.01-0.3 之间
- **树深度**: 一般设置为 3-10，过深容易过拟合
- **树数量**: 根据数据集大小和复杂度调整，可以通过早停机制优化
- **正则化**: lambda 和 gamma 用于控制过拟合

### 2. GPU 内存管理

- 根据 GPU 内存大小调整 `max_num_bin` 参数
- 大数据集可以考虑使用多 GPU 并行训练
- 监控 GPU 内存使用情况

### 3. 数据预处理

- 确保数据格式正确（LibSVM 格式）
- 处理缺失值
- 特征缩放（如果必要）

## 常见问题 (Common Issues)

### 1. GPU 内存不足

**问题**: CUDA out of memory
**解决方案**: 
- 减少 `max_num_bin` 参数
- 使用更少的 GPU 设备
- 减少批处理大小

### 2. 编译错误

**问题**: CUDA 版本不匹配
**解决方案**:
- 确保 CUDA 版本 >= 11.7
- 在 CMakeLists.txt 中设置正确的架构参数

### 3. 性能优化

**建议**:
- 使用 `tree_method=hist` 获得更好的性能
- 调整 `max_num_bin` 平衡速度和精度
- 合理设置 GPU 数量

## 配置文件示例 (Configuration File Example)

可以创建配置文件来管理参数：

```
# config.conf
data=dataset/adult
objective=binary:logistic
tree_method=hist
n_trees=100
depth=6
learning_rate=0.1
lambda=1.0
gamma=0.0
max_num_bin=255
n_gpus=1
verbose=1
```

使用配置文件：
```bash
./bin/scalagbm-train config.conf
```

## 总结 (Summary)

ScalaGBM 的模型初始化包括参数设置、GPU 设备管理、内存分配和基础分数计算等步骤。通过合理配置参数和使用最佳实践，可以获得高效的训练性能和良好的模型效果。