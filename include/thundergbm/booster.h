
#ifndef THUNDERGBM_BOOSTER_H
#define THUNDERGBM_BOOSTER_H

#include <thundergbm/objective/objective_function.h>
#include <thundergbm/metric/metric.h>
#include <thundergbm/builder/function_builder.h>
#include <thundergbm/util/multi_device.h>
#include "thundergbm/common.h"
#include "syncarray.h"
#include "tree.h"
#include "row_sampler.h"

std::mutex mtx;
class Booster {
public:
    void init(const DataSet &dataSet, GBMParam &param);

    void boost(vector<vector<Tree>> &boosted_model,int epoch,int total_epoch);

private:
    MSyncArray<GHPair> gradients;
    std::unique_ptr<ObjectiveFunction> obj;
    std::unique_ptr<Metric> metric;
    MSyncArray<float_type> y;
    std::unique_ptr<FunctionBuilder> fbuilder;
    RowSampler rowSampler;
    GBMParam param;
    int n_devices;
};

void Booster::init(const DataSet &dataSet, GBMParam &param) {
    // 第一步：GPU设备管理和验证 (Step 1: GPU device management and validation)
    // 检查可用的GPU设备数量，确保有足够的设备进行训练
    // (Check available GPU devices to ensure sufficient hardware for training)
    int n_available_device;
    cudaGetDeviceCount(&n_available_device);
    CHECK_GE(n_available_device, param.n_device) << "only " << n_available_device
                                            << " GPUs available; please set correct number of GPUs to use";
    this->param = param;
    
    // 第二步：初始化函数构建器 (Step 2: Initialize function builder)
    // 根据tree_method参数选择合适的树构建算法（hist或exact）
    // (Select appropriate tree building algorithm based on tree_method parameter)
    fbuilder.reset(FunctionBuilder::create(param.tree_method));
    fbuilder->init(dataSet, param);
    
    // 第三步：初始化目标函数 (Step 3: Initialize objective function)
    // 根据objective参数创建相应的损失函数（如binary:logistic, reg:linear等）
    // (Create corresponding loss function based on objective parameter)
    obj.reset(ObjectiveFunction::create(param.objective));
    obj->configure(param, dataSet);
    
    // 第四步：初始化评估指标 (Step 4: Initialize evaluation metrics)
    // 根据目标函数自动选择合适的评估指标
    // (Automatically select appropriate evaluation metrics based on objective function)
    metric.reset(Metric::create(obj->default_metric_name()));
    metric->configure(param, dataSet);

    // 第五步：分配GPU内存 (Step 5: Allocate GPU memory)
    // 为梯度和预测值分配多GPU内存空间
    // (Allocate multi-GPU memory space for gradients and predictions)
    n_devices = param.n_device;
    int n_outputs = param.num_class * dataSet.n_instances();
    gradients = MSyncArray<GHPair>(n_devices, n_outputs);
    y = MSyncArray<float_type>(n_devices, dataSet.n_instances());

    // 第六步：数据传输到GPU (Step 6: Transfer data to GPU)
    // 将训练标签从CPU内存复制到所有GPU设备
    // (Copy training labels from CPU memory to all GPU devices)
    DO_ON_MULTI_DEVICES(n_devices, [&](int device_id) {
        y[device_id].copy_from(dataSet.y.data(), dataSet.n_instances());
    });

    // 第七步：初始化基础分数 (Step 7: Initialize base score)
    // 计算初始预测值，这是模型训练的起点
    // (Calculate initial prediction values, this is the starting point for model training)
    //TODO no need to calculate many times
    DO_ON_MULTI_DEVICES(n_devices, [&](int device_id){
        param.base_score = obj->init_base_score(y[device_id], fbuilder->get_raw_y_predict()[device_id], gradients[device_id]);
    });
}

void Booster::boost(vector<vector<Tree>> &boosted_model,int epoch,int total_epoch) {
    TIMED_FUNC(timerObj);
    std::unique_lock<std::mutex> lock(mtx);
    //update gradients
    DO_ON_MULTI_DEVICES(n_devices, [&](int device_id) {
        //for(int i =0;i<5;i++){
        //    LOG(INFO)<<"prediction index is "<<i<<" value is "<<fbuilder->get_y_predict()[device_id].host_data()[i];
        //}
        obj->get_gradient(y[device_id], fbuilder->get_y_predict()[device_id], gradients[device_id]);
    });
    if (param.bagging) rowSampler.do_bagging(gradients);
    PERFORMANCE_CHECKPOINT(timerObj);
    //build new model/approximate function
    boosted_model.push_back(fbuilder->build_approximate(gradients));

    PERFORMANCE_CHECKPOINT(timerObj);
    //show metric on training set
    auto res =  metric->get_score(fbuilder->get_y_predict().front());
    LOG(INFO) <<"["<<epoch<<"/"<<total_epoch<<"] "<< metric->get_name() << " = " <<res<<" acc = "<< (1-res);
}

#endif //THUNDERGBM_BOOSTER_H
