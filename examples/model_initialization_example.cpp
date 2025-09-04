/**
 * ScalaGBM 模型初始化示例代码
 * Example code for ScalaGBM model initialization
 */

#include <thundergbm/trainer.h>
#include <thundergbm/parser.h>
#include <thundergbm/dataset.h>

/**
 * 创建并初始化GBM参数的示例函数
 * Example function to create and initialize GBM parameters
 */
GBMParam create_default_params() {
    GBMParam param;
    
    // 基础参数设置 (Basic parameter settings)
    param.depth = 6;                        // 树深度 (Tree depth)
    param.n_trees = 100;                    // 树数量 (Number of trees)
    param.learning_rate = 0.1;              // 学习率 (Learning rate)
    param.n_device = 1;                     // GPU设备数量 (Number of GPU devices)
    
    // 正则化参数 (Regularization parameters)
    param.lambda = 1.0;                     // L2正则化 (L2 regularization)
    param.gamma = 0.0;                      // 最小分割损失 (Minimum split loss)
    param.min_child_weight = 1;             // 最小子节点权重 (Minimum child weight)
    
    // 树构建参数 (Tree building parameters)
    param.tree_method = "hist";             // 树构建方法 (Tree building method)
    param.max_num_bin = 255;                // 最大分桶数 (Maximum number of bins)
    param.column_sampling_rate = 1.0;       // 列采样率 (Column sampling rate)
    
    // 目标函数设置 (Objective function settings)
    param.objective = "binary:logistic";    // 目标函数 (Objective function)
    param.num_class = 2;                    // 类别数量 (Number of classes)
    param.tree_per_rounds = 1;              // 每轮树数量 (Trees per round)
    
    // 其他设置 (Other settings)
    param.verbose = 1;                      // 日志级别 (Verbose level)
    param.profiling = false;                // 性能分析 (Profiling)
    param.bagging = false;                  // 是否使用bagging (Whether to use bagging)
    param.n_parallel_trees = 1;             // 并行树数量 (Number of parallel trees)
    param.base_score = 0.0;                 // 基础分数 (Base score)
    
    return param;
}

/**
 * 二分类任务参数配置示例
 * Example parameter configuration for binary classification
 */
GBMParam create_binary_classification_params() {
    GBMParam param = create_default_params();
    
    param.objective = "binary:logistic";
    param.num_class = 2;
    param.tree_per_rounds = 1;
    param.learning_rate = 0.1;
    param.depth = 6;
    param.n_trees = 100;
    
    return param;
}

/**
 * 多分类任务参数配置示例
 * Example parameter configuration for multi-class classification
 */
GBMParam create_multiclass_params(int num_classes) {
    GBMParam param = create_default_params();
    
    param.objective = "multi:softmax";
    param.num_class = num_classes;
    param.tree_per_rounds = num_classes;  // 每轮为每个类别构建一棵树
    param.learning_rate = 0.1;
    param.depth = 6;
    param.n_trees = 100;
    
    return param;
}

/**
 * 回归任务参数配置示例
 * Example parameter configuration for regression
 */
GBMParam create_regression_params() {
    GBMParam param = create_default_params();
    
    param.objective = "reg:linear";
    param.num_class = 1;
    param.tree_per_rounds = 1;
    param.learning_rate = 0.1;
    param.depth = 6;
    param.n_trees = 100;
    
    return param;
}

/**
 * 高性能配置示例 (适用于大数据集)
 * High-performance configuration example (for large datasets)
 */
GBMParam create_high_performance_params() {
    GBMParam param = create_default_params();
    
    param.tree_method = "hist";             // 使用直方图方法提高速度
    param.max_num_bin = 128;                // 减少分桶数以节省内存
    param.column_sampling_rate = 0.8;       // 列采样以减少计算量
    param.learning_rate = 0.05;             // 较小的学习率
    param.n_trees = 200;                    // 更多的树来补偿较小的学习率
    param.depth = 8;                        // 稍深的树
    param.lambda = 2.0;                     // 更强的正则化
    param.gamma = 0.1;
    
    return param;
}

/**
 * 内存优化配置示例 (适用于GPU内存有限的情况)
 * Memory-optimized configuration example (for limited GPU memory)
 */
GBMParam create_memory_optimized_params() {
    GBMParam param = create_default_params();
    
    param.max_num_bin = 64;                 // 大幅减少分桶数
    param.depth = 4;                        // 较浅的树
    param.column_sampling_rate = 0.6;       // 更激进的列采样
    param.n_device = 1;                     // 使用单GPU
    param.learning_rate = 0.15;             // 稍高的学习率以补偿较浅的树
    param.n_trees = 150;
    
    return param;
}

/**
 * 完整的模型初始化和训练示例
 * Complete model initialization and training example
 */
void train_model_example(const std::string& data_path, 
                        const std::string& task_type = "binary") {
    // 1. 创建参数 (Create parameters)
    GBMParam param;
    if (task_type == "binary") {
        param = create_binary_classification_params();
    } else if (task_type == "multiclass") {
        param = create_multiclass_params(7);  // 假设7个类别
    } else if (task_type == "regression") {
        param = create_regression_params();
    } else {
        param = create_default_params();
    }
    
    // 设置数据路径 (Set data path)
    param.path = data_path;
    
    // 2. 加载数据集 (Load dataset)
    DataSet dataset;
    dataset.load_from_file(param.path, param);
    
    // 3. 创建训练器并初始化 (Create trainer and initialize)
    TreeTrainer trainer;
    
    // 4. 开始训练 (Start training)
    std::cout << "开始训练模型... (Starting model training...)" << std::endl;
    std::cout << "参数配置: (Parameter configuration:)" << std::endl;
    std::cout << "  目标函数: " << param.objective << std::endl;
    std::cout << "  树数量: " << param.n_trees << std::endl;
    std::cout << "  树深度: " << param.depth << std::endl;
    std::cout << "  学习率: " << param.learning_rate << std::endl;
    
    vector<vector<Tree>> model = trainer.train(param, dataset);
    
    // 5. 保存模型 (Save model)
    Parser parser;
    parser.save_model("trained_model.model", param, model, dataset);
    
    std::cout << "模型训练完成并已保存! (Model training completed and saved!)" << std::endl;
}

/**
 * 参数验证函数
 * Parameter validation function
 */
bool validate_params(const GBMParam& param) {
    // 基本参数检查 (Basic parameter checks)
    if (param.depth <= 0 || param.depth > 20) {
        std::cerr << "错误: 树深度应在1-20之间 (Error: Tree depth should be between 1-20)" << std::endl;
        return false;
    }
    
    if (param.n_trees <= 0) {
        std::cerr << "错误: 树数量应大于0 (Error: Number of trees should be > 0)" << std::endl;
        return false;
    }
    
    if (param.learning_rate <= 0 || param.learning_rate > 1) {
        std::cerr << "错误: 学习率应在(0,1]范围内 (Error: Learning rate should be in (0,1])" << std::endl;
        return false;
    }
    
    if (param.n_device <= 0) {
        std::cerr << "错误: GPU设备数应大于0 (Error: Number of devices should be > 0)" << std::endl;
        return false;
    }
    
    // 目标函数检查 (Objective function checks)
    if (param.objective.find("multi:") != std::string::npos && param.num_class <= 2) {
        std::cerr << "错误: 多分类任务的类别数应大于2 (Error: Multi-class should have > 2 classes)" << std::endl;
        return false;
    }
    
    if (param.objective.find("binary:") != std::string::npos && param.num_class != 2) {
        std::cerr << "警告: 二分类任务建议设置num_class=2 (Warning: Binary classification should set num_class=2)" << std::endl;
    }
    
    return true;
}

/**
 * 打印参数配置信息
 * Print parameter configuration information
 */
void print_param_info(const GBMParam& param) {
    std::cout << "=== ScalaGBM 参数配置 (Parameter Configuration) ===" << std::endl;
    std::cout << "基础参数 (Basic Parameters):" << std::endl;
    std::cout << "  树深度 (Tree Depth): " << param.depth << std::endl;
    std::cout << "  树数量 (Number of Trees): " << param.n_trees << std::endl;
    std::cout << "  学习率 (Learning Rate): " << param.learning_rate << std::endl;
    std::cout << "  GPU设备数 (GPU Devices): " << param.n_device << std::endl;
    
    std::cout << "模型参数 (Model Parameters):" << std::endl;
    std::cout << "  目标函数 (Objective): " << param.objective << std::endl;
    std::cout << "  类别数 (Number of Classes): " << param.num_class << std::endl;
    std::cout << "  树构建方法 (Tree Method): " << param.tree_method << std::endl;
    std::cout << "  最大分桶数 (Max Bins): " << param.max_num_bin << std::endl;
    
    std::cout << "正则化参数 (Regularization Parameters):" << std::endl;
    std::cout << "  Lambda (L2): " << param.lambda << std::endl;
    std::cout << "  Gamma (Min Split Loss): " << param.gamma << std::endl;
    std::cout << "  最小子节点权重 (Min Child Weight): " << param.min_child_weight << std::endl;
    
    std::cout << "采样参数 (Sampling Parameters):" << std::endl;
    std::cout << "  列采样率 (Column Sampling Rate): " << param.column_sampling_rate << std::endl;
    std::cout << "  是否Bagging (Bagging): " << (param.bagging ? "是" : "否") << std::endl;
    std::cout << "=============================================" << std::endl;
}