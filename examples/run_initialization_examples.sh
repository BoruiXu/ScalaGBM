#!/bin/bash

# ScalaGBM 模型初始化示例脚本
# ScalaGBM Model Initialization Example Script

echo "=== ScalaGBM 模型初始化示例 (ScalaGBM Model Initialization Examples) ==="

# 检查可执行文件是否存在
if [ ! -f "../bin/scalagbm-train" ]; then
    echo "错误: 找不到 scalagbm-train 可执行文件"
    echo "Error: scalagbm-train executable not found"
    echo "请先编译项目: mkdir build && cd build && cmake .. && make -j"
    echo "Please compile the project first: mkdir build && cd build && cmake .. && make -j"
    exit 1
fi

echo ""
echo "1. 二分类任务示例 (Binary Classification Example)"
echo "使用默认参数进行二分类训练 (Binary classification with default parameters)"
echo "命令 (Command):"
echo "../bin/scalagbm-train data=../dataset/adult objective=binary:logistic tree_method=hist n_trees=40 depth=6"
echo ""

echo "2. 多分类任务示例 (Multi-class Classification Example)"
echo "7类分类任务 (7-class classification task)"
echo "命令 (Command):"
echo "../bin/scalagbm-train data=../dataset/covtype objective=multi:softmax num_class=7 tree_method=hist n_trees=100 depth=6"
echo ""

echo "3. 回归任务示例 (Regression Example)"
echo "线性回归任务 (Linear regression task)"
echo "命令 (Command):"
echo "../bin/scalagbm-train data=../dataset/housing objective=reg:linear tree_method=hist n_trees=100 depth=6"
echo ""

echo "4. 高性能配置示例 (High Performance Configuration Example)"
echo "适用于大数据集的优化配置 (Optimized configuration for large datasets)"
echo "命令 (Command):"
echo "../bin/scalagbm-train data=../dataset/large_dataset objective=binary:logistic tree_method=hist max_num_bin=128 column_sampling_rate=0.8 learning_rate=0.05 n_trees=200 depth=8 lambda=2.0"
echo ""

echo "5. 内存优化配置示例 (Memory Optimized Configuration Example)"  
echo "适用于GPU内存有限的情况 (For limited GPU memory scenarios)"
echo "命令 (Command):"
echo "../bin/scalagbm-train data=../dataset/dataset objective=binary:logistic tree_method=hist max_num_bin=64 depth=4 column_sampling_rate=0.6 learning_rate=0.15 n_trees=150"
echo ""

echo "6. 使用配置文件示例 (Using Configuration File Example)"
echo "通过配置文件管理参数 (Manage parameters through configuration file)"
echo "命令 (Command):"
echo "../bin/scalagbm-train config_template.conf"
echo ""

echo "7. 混合参数配置示例 (Mixed Parameter Configuration Example)"
echo "配置文件 + 命令行参数覆盖 (Configuration file + command line override)"
echo "命令 (Command):"
echo "../bin/scalagbm-train config_template.conf learning_rate=0.05 n_trees=200 verbose=2"
echo ""

echo "=== 参数说明 (Parameter Descriptions) ==="
echo ""
echo "基础参数 (Basic Parameters):"
echo "  data              : 训练数据路径 (Training data path)"
echo "  objective         : 目标函数 (Objective function)"
echo "  tree_method       : 树构建方法 (Tree building method)"
echo "  n_trees          : 树数量 (Number of trees)"
echo "  depth            : 树深度 (Tree depth)"
echo "  learning_rate    : 学习率 (Learning rate)"
echo ""
echo "性能参数 (Performance Parameters):"
echo "  max_num_bin      : 最大分桶数 (Maximum number of bins)"
echo "  n_gpus           : GPU数量 (Number of GPUs)"
echo "  column_sampling_rate : 列采样率 (Column sampling rate)"
echo ""
echo "正则化参数 (Regularization Parameters):"
echo "  lambda           : L2正则化 (L2 regularization)"
echo "  gamma            : 最小分割损失 (Minimum split loss)"
echo "  min_child_weight : 最小子节点权重 (Minimum child weight)"
echo ""

echo "=== 目标函数类型 (Objective Function Types) ==="
echo "  binary:logistic  : 二分类逻辑回归 (Binary logistic regression)"
echo "  multi:softmax    : 多分类 (Multi-class classification)"
echo "  multi:softprob   : 多分类概率 (Multi-class with probabilities)"
echo "  reg:linear       : 线性回归 (Linear regression)"
echo "  reg:logistic     : 逻辑回归 (Logistic regression)"
echo ""

echo "=== 更多信息 (More Information) ==="
echo "详细文档: docs/model_initialization.md (Detailed documentation: docs/model_initialization.md)"
echo "示例代码: examples/model_initialization_example.cpp (Example code: examples/model_initialization_example.cpp)"
echo "配置模板: examples/config_template.conf (Configuration template: examples/config_template.conf)"
echo ""

# 创建一个简单的测试配置文件
echo "=== 创建测试配置文件 (Creating Test Configuration File) ==="
cat > test_config.conf << EOF
# 测试配置文件 (Test configuration file)
data=../dataset/test_dataset.txt
objective=binary:logistic
tree_method=hist
n_trees=10
depth=4
learning_rate=0.1
verbose=1
EOF

echo "已创建测试配置文件: test_config.conf (Created test configuration file: test_config.conf)"
echo "运行测试: ../bin/scalagbm-train test_config.conf (Run test: ../bin/scalagbm-train test_config.conf)"