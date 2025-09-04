
#include "thundergbm/parser.h"
#include <cstring>
using namespace std;

void Parser::parse_param(GBMParam &model_param, int argc, char **argv){
    // 初始化默认参数值 (Initialize default parameter values)
    // 这些默认值适合大部分场景，可以根据具体需求调整
    // (These default values work for most scenarios, can be adjusted based on specific needs)
    
    // 树结构参数 (Tree structure parameters)
    model_param.depth = 6;                    // 默认树深度 (Default tree depth)
    model_param.n_trees = 40;                 // 默认迭代轮数 (Default number of iterations)
    
    // 设备和性能参数 (Device and performance parameters)
    model_param.n_device = 1;                 // 默认使用1个GPU (Default: use 1 GPU)
    model_param.max_num_bin = 255;            // 直方图分桶数，影响内存使用和精度 (Histogram bins, affects memory and accuracy)
    model_param.verbose = 1;                  // 日志详细程度 (Logging verbosity)
    model_param.profiling = false;            // 性能分析开关 (Performance profiling toggle)
    
    // 正则化参数 (Regularization parameters)
    model_param.min_child_weight = 1;         // 子节点最小权重，防止过拟合 (Min child weight, prevents overfitting)
    model_param.lambda = 1;                   // L2正则化参数 (L2 regularization parameter)
    model_param.gamma = 1;                    // 最小分割损失，控制树的复杂度 (Min split loss, controls tree complexity)
    model_param.rt_eps = 1e-6;                // 数值计算精度 (Numerical precision)
    
    // 采样参数 (Sampling parameters)
    model_param.column_sampling_rate = 1;     // 列采样率，1表示使用所有特征 (Column sampling rate, 1 = use all features)
    model_param.bagging = false;              // 是否使用bagging (Whether to use bagging)
    model_param.n_parallel_trees = 1;         // 并行树数量 (Number of parallel trees)
    
    // 学习参数 (Learning parameters)
    model_param.learning_rate = 1;            // 学习率，控制每棵树的贡献 (Learning rate, controls each tree's contribution)
    model_param.objective = "reg:linear";     // 默认目标函数：线性回归 (Default objective: linear regression)
    model_param.num_class = 1;                // 目标类别数 (Number of target classes)
    
    // 数据和算法参数 (Data and algorithm parameters)
    model_param.path = "../dataset/test_dataset.txt";  // 默认数据路径 (Default data path)
    model_param.tree_method = "auto";         // 树构建方法：auto会自动选择最适合的方法 (Tree building method: auto selects best method)
    model_param.tree_per_rounds = 1;          // 每轮构建的树数量，取决于分类数 (Trees per round, depends on number of classes)

    if (argc < 2) {
        printf("Usage: <config>\n");
        exit(0);
    }

    //parsing parameter values from configuration file or command line
    auto parse_value = [&](const char *name_val){
        char name[256], val[256];
        if (sscanf(name_val, "%[^=]=%s", name, val) == 2) {
            string str_name(name);
            if((str_name.compare("max_depth") == 0) || (str_name.compare("depth") == 0))
                model_param.depth = atoi(val);
            else if((str_name.compare("num_round") == 0) || (str_name.compare("n_trees") == 0))
                model_param.n_trees = atoi(val);
            else if(str_name.compare("n_gpus") == 0)
                model_param.n_device = atoi(val);
            else if((str_name.compare("verbosity") == 0) || (str_name.compare("verbose") == 0))
                model_param.verbose = atoi(val);
            else if(str_name.compare("profiling") == 0)
                model_param.profiling = atoi(val);
            else if(str_name.compare("data") == 0)
                model_param.path = val;
            else if((str_name.compare("max_bin") == 0) || (str_name.compare("max_num_bin") == 0))
                model_param.max_num_bin = atoi(val);
            else if((str_name.compare("colsample") == 0) || (str_name.compare("column_sampling_rate") == 0))
                model_param.column_sampling_rate = atof(val);
            else if(str_name.compare("bagging") == 0)
                model_param.bagging = atoi(val);
            else if((str_name.compare("num_parallel_tree") == 0) || (str_name.compare("n_parallel_trees") == 0))
                model_param.n_parallel_trees = atoi(val);
            else if(str_name.compare("eta") == 0 || str_name.compare("learning_rate") == 0)
                model_param.learning_rate = atof(val);
            else if(str_name.compare("objective") == 0)
                model_param.objective = val;
            else if(str_name.compare("num_class") == 0)
                model_param.num_class = atoi(val);
            else if(str_name.compare("min_child_weight") == 0)
                model_param.min_child_weight = atoi(val);
            else if(str_name.compare("lambda") == 0 || str_name.compare("lambda_tgbm") == 0)
                model_param.lambda = atof(val);
            else if(str_name.compare("gamma") == 0 || str_name.compare("min_split_loss") == 0)
                model_param.gamma = atof(val);
            else if(str_name.compare("tree_method") == 0)
                model_param.tree_method = val;
            else
                LOG(INFO) << "\"" << name << "\" is unknown option!";
        }
        else{
            string str_name(name);
            if(str_name.compare("-help") == 0){
                printf("please refer to \"docs/parameters.md\" in the GitHub repository for more information about setting the options\n");
                exit(0);
            }
        }

    };

    //read configuration file
    std::ifstream conf_file(argv[1]);
    std::string line;
    while (std::getline(conf_file, line))
    {
        //LOG(INFO) << line;
        parse_value(line.c_str());
    }

    //TODO: confirm handling spaces around "="
    for (int i = 0; i < argc; ++i) {
        parse_value(argv[i]);
    }//end parsing parameters
}

void Parser::load_model(string model_path, GBMParam &model_param, vector<vector<Tree>> &boosted_model, DataSet & dataset) {
    std::ifstream ifs(model_path, ios::binary);
    CHECK_EQ(ifs.is_open(), true);
    int length;
    ifs.read((char*)&length, sizeof(length));
    char * temp = new char[length+1];
    temp[length] = '\0';
    // read param.objective
    ifs.read(temp, length);
    string str(temp);
    model_param.objective = str;
    ifs.read((char*)&model_param.learning_rate, sizeof(model_param.learning_rate));
    ifs.read((char*)&model_param.num_class, sizeof(model_param.num_class));
    ifs.read((char*)&model_param.n_trees, sizeof(model_param.n_trees));
    ifs.read((char*)&model_param.base_score, sizeof(model_param.base_score));
    int label_size;
    ifs.read((char*)&label_size, sizeof(label_size));
    float_type f;
    dataset.label.clear();
    for (int i = 0; i < label_size; ++i) {
        ifs.read((char*)&f, sizeof(float_type));
        dataset.label.push_back(f);
    }
    int boosted_model_size;
    ifs.read((char*)&boosted_model_size, sizeof(boosted_model_size));
    Tree t;
    vector<Tree> v;
    for (int i = 0; i < boosted_model_size; ++i) {
        int boost_model_i_size;
        ifs.read((char*)&boost_model_i_size, sizeof(boost_model_i_size));
        for (int j = 0; j < boost_model_i_size; ++j) {
            size_t syn_node_size;
            ifs.read((char*)&syn_node_size, sizeof(syn_node_size));
            SyncArray<Tree::TreeNode> tmp(syn_node_size);
            ifs.read((char*)tmp.host_data(), sizeof(Tree::TreeNode) * syn_node_size);
            t.nodes.resize(tmp.size());
            t.nodes.copy_from(tmp);
            v.push_back(t);
        }
        boosted_model.push_back(v);
        v.clear();
    }
    ifs.close();
}
void Parser::save_model(string model_path, GBMParam &model_param, vector<vector<Tree>> &boosted_model, DataSet &dataset) {
    ofstream out_model_file(model_path, ios::binary);
    CHECK_EQ(out_model_file.is_open(), true);
    int length = model_param.objective.length();
    out_model_file.write((char*)&length, sizeof(length));
    out_model_file.write(model_param.objective.c_str(), model_param.objective.length());
    out_model_file.write((char*)&model_param.learning_rate, sizeof(model_param.learning_rate));
    out_model_file.write((char*)&model_param.num_class, sizeof(model_param.num_class));
    out_model_file.write((char*)&model_param.n_trees, sizeof(model_param.n_trees));
    out_model_file.write((char*)&model_param.base_score, sizeof(model_param.base_score));
    int label_size = dataset.label.size();
    out_model_file.write((char*)&label_size, sizeof(label_size));
    out_model_file.write((char*)&dataset.label[0], dataset.label.size() * sizeof(float_type));
    int boosted_model_size = boosted_model.size();
    out_model_file.write((char*)&boosted_model_size, sizeof(boosted_model_size));
    for(int j = 0; j < boosted_model.size(); ++j) {
        int boosted_model_j_size = boosted_model[j].size();
        out_model_file.write((char*)&boosted_model_j_size, sizeof(boosted_model_j_size));
        for (int i = 0; i < boosted_model_j_size; ++i) {
            size_t syn_node_size = boosted_model[j][i].nodes.size();
            out_model_file.write((char*)&syn_node_size, sizeof(syn_node_size));
            out_model_file.write((char*)boosted_model[j][i].nodes.host_data(), syn_node_size * sizeof(Tree::TreeNode));
        }
    }
    out_model_file.close();
}
