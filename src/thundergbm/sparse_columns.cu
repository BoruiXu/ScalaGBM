//
// Created by shijiashuai on 5/7/18.
//
#include <thundergbm/sparse_columns.h>
#include <thundergbm/util/cub_wrapper.h>

#include "cusparse.h"
#include "omp.h"
#include "thundergbm/sparse_columns.h"
#include "thundergbm/util/device_lambda.cuh"
#include "thundergbm/util/multi_device.h"

#include <chrono>
#include<iostream>
typedef std::chrono::high_resolution_clock Clock;
#define TDEF(x_) std::chrono::high_resolution_clock::time_point x_##_t0, x_##_t1;
#define TSTART(x_) x_##_t0 = Clock::now();
#define TEND(x_) x_##_t1 = Clock::now();
#define TPRINT(x_, str) printf("%-20s \t%.6f\t sec\n", str, std::chrono::duration_cast<std::chrono::microseconds>(x_##_t1 - x_##_t0).count()/1e6);
#define TINT(x_) std::chrono::duration_cast<std::chrono::microseconds>(x_##_t1 - x_##_t0).count()
//#define CSC
// FIXME remove this function
void correct_start(int *csc_col_ptr_2d_data, int first_col_start,
                   int n_column_sub) {
    device_loop(n_column_sub + 1, [=] __device__(int col_id) {
        csc_col_ptr_2d_data[col_id] =
            csc_col_ptr_2d_data[col_id] - first_col_start;
    });
};
void SparseColumns::csr2csc_gpu(
    const DataSet &dataset, vector<std::unique_ptr<SparseColumns>> &v_columns) {
    LOG(INFO) << "convert csr to csc using gpu...";
    std::chrono::high_resolution_clock timer;
    auto t_start = timer.now();

    // three arrays (on GPU/CPU) for csr representation
    //use arry in device 0 to store
    this->column_offset = 0;
    SparseColumns &columns = *v_columns[0];

    
    columns.csr_val.resize(dataset.csr_val.size());
    columns.csr_col_idx.resize(dataset.csr_col_idx.size());
    columns.csr_row_ptr.resize(dataset.csr_row_ptr.size());

    // copy data to the three arrays
    columns.csr_val.copy_from(dataset.csr_val.data(), columns.csr_val.size());
    columns.csr_col_idx.copy_from(dataset.csr_col_idx.data(), columns.csr_col_idx.size());
    columns.csr_row_ptr.copy_from(dataset.csr_row_ptr.data(), columns.csr_row_ptr.size());
    
    n_column = dataset.n_features_;
    n_row = dataset.n_instances();
    nnz = dataset.csr_val.size();

#ifdef CSC    
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);

    columns.csc_val_origin.resize(nnz);
    columns.csc_row_idx_origin.resize(nnz);
    columns.csc_col_ptr_origin.resize(n_column + 1);

#if (CUDART_VERSION >= 11000)
#ifdef USE_DOUBLE
    cudaDataType data_type = CUDA_R_64F;
#else
    cudaDataType data_type = CUDA_R_32F;
#endif
    // TODO fix the issue of < cuda9
    size_t buffer_size = 0;
    cusparseCsr2cscEx2_bufferSize(
        handle, dataset.n_instances(), n_column, nnz, 
        columns.csr_val.device_data(),columns.csr_row_ptr.device_data(), columns.csr_col_idx.device_data(), 
        columns.csc_val_origin.device_data(),columns.csc_col_ptr_origin.device_data(), columns.csc_row_idx_origin.device_data(), 
        data_type,CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1, &buffer_size);

    SyncArray<char> tmp_buffer(buffer_size);
    
    cusparseCsr2cscEx2(
        handle, dataset.n_instances(), n_column, nnz, 
        columns.csr_val.device_data(), columns.csr_row_ptr.device_data(), columns.csr_col_idx.device_data(), 
        columns.csc_val_origin.device_data(), columns.csc_col_ptr_origin.device_data(), columns.csc_row_idx_origin.device_data(), 
        data_type, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1, tmp_buffer.device_data());
#else
    cusparseScsr2csc(handle, dataset.n_instances(), n_column, nnz,
                     val.device_data(), row_ptr.device_data(),
                     col_idx.device_data(), csc_val.device_data(),
                     csc_row_idx.device_data(), csc_col_ptr.device_data(),
                     CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
#endif
    cudaDeviceSynchronize();
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descr);

    
    tmp_buffer.resize(0);
    SyncMem::clear_cache();
    int gpu_num;
    cudaError_t err = cudaGetDeviceCount(&gpu_num);
    std::atexit([]() { SyncMem::clear_cache(); });
#endif
    int n_device = v_columns.size();
    //int ave_n_columns = n_column / n_device;
    DO_ON_MULTI_DEVICES(n_device, [&](int device_id) {
        SparseColumns &columns = *v_columns[device_id];

        int first_col_id = 0;
        //int first_col_start = 0;
        int n_column_sub = n_column;
        int nnz_sub = nnz; 

        //FIXME bug, change varibale name to origin
        if(n_device>1){
        }
        else{
            columns.column_offset = first_col_id + this->column_offset;
            columns.nnz = nnz_sub;
            columns.n_column = n_column_sub;
            columns.n_row = n_row;
        }
    });
    //SyncMem::clear_cache();
    auto t_end = timer.now();
    std::chrono::duration<float> used_time = t_end - t_start;
    LOG(INFO) << "Converting csr to csc using time: " << used_time.count()
              << " s";
}
//void SparseColumns::csr2csc_gpu(
//    const DataSet &dataset, vector<std::unique_ptr<SparseColumns>> &v_columns) {
//    LOG(INFO) << "convert csr to csc using gpu...";
//    std::chrono::high_resolution_clock timer;
//    auto t_start = timer.now();
//
//    // three arrays (on GPU/CPU) for csr representation
//    this->column_offset = 0;
//    SyncArray<float_type> val;
//    SyncArray<int> col_idx;
//    SyncArray<int> row_ptr;
//    val.resize(dataset.csr_val.size());
//    col_idx.resize(dataset.csr_col_idx.size());
//    row_ptr.resize(dataset.csr_row_ptr.size());
//
//    // copy data to the three arrays
//    val.copy_from(dataset.csr_val.data(), val.size());
//    col_idx.copy_from(dataset.csr_col_idx.data(), col_idx.size());
//    row_ptr.copy_from(dataset.csr_row_ptr.data(), row_ptr.size());
//    cusparseHandle_t handle;
//    cusparseMatDescr_t descr;
//    cusparseCreate(&handle);
//    cusparseCreateMatDescr(&descr);
//    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
//    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
//
//    n_column = dataset.n_features_;
//    n_row = dataset.n_instances();
//    nnz = dataset.csr_val.size();
//    csc_val.resize(nnz);
//    csc_row_idx.resize(nnz);
//    csc_col_ptr.resize(n_column + 1);
//
//#if (CUDART_VERSION >= 11000)
//#ifdef USE_DOUBLE
//    cudaDataType data_type = CUDA_R_64F;
//#else
//    cudaDataType data_type = CUDA_R_32F;
//#endif
//    // TODO fix the issue of < cuda9
//    size_t buffer_size = 0;
//    cusparseCsr2cscEx2_bufferSize(
//        handle, dataset.n_instances(), n_column, nnz, val.device_data(),
//        row_ptr.device_data(), col_idx.device_data(), csc_val.device_data(),
//        csc_col_ptr.device_data(), csc_row_idx.device_data(), data_type,
//        CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
//        CUSPARSE_CSR2CSC_ALG1, &buffer_size);
//    SyncArray<char> tmp_buffer(buffer_size);
//    cusparseCsr2cscEx2(
//        handle, dataset.n_instances(), n_column, nnz, val.device_data(),
//        row_ptr.device_data(), col_idx.device_data(), csc_val.device_data(),
//        csc_col_ptr.device_data(), csc_row_idx.device_data(), data_type,
//        CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
//        CUSPARSE_CSR2CSC_ALG1, tmp_buffer.device_data());
//#else
//    cusparseScsr2csc(handle, dataset.n_instances(), n_column, nnz,
//                     val.device_data(), row_ptr.device_data(),
//                     col_idx.device_data(), csc_val.device_data(),
//                     csc_row_idx.device_data(), csc_col_ptr.device_data(),
//                     CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
//#endif
//    cudaDeviceSynchronize();
//    cusparseDestroy(handle);
//    cusparseDestroyMatDescr(descr);
//
//    //val.resize(0);
//    //row_ptr.resize(0);
//    //col_idx.resize(0);
//    tmp_buffer.resize(0);
//    SyncMem::clear_cache();
//    int gpu_num;
//    cudaError_t err = cudaGetDeviceCount(&gpu_num);
//    std::atexit([]() { SyncMem::clear_cache(); });
//
//    int n_device = v_columns.size();
//    int ave_n_columns = n_column / n_device;
//    DO_ON_MULTI_DEVICES(n_device, [&](int device_id) {
//        SparseColumns &columns = *v_columns[device_id];
//
//        int first_col_id = 0;
//        int first_col_start = 0;
//        int n_column_sub = csc_col_ptr.size()-1;
//        int nnz_sub = csc_val.size();
//        if(n_device>1){
//            const int *csc_col_ptr_data = csc_col_ptr.host_data();
//            first_col_id = device_id * ave_n_columns;
//            n_column_sub = (device_id < n_device - 1) ? ave_n_columns
//                                                      : n_column - first_col_id;
//            first_col_start = csc_col_ptr_data[first_col_id];
//            nnz_sub = (device_id < n_device - 1)
//                          ? (csc_col_ptr_data[(device_id + 1) * ave_n_columns] -
//                             first_col_start)
//                          : (nnz - first_col_start);
//        }
//        columns.column_offset = first_col_id + this->column_offset;
//        columns.nnz = nnz_sub;
//        columns.n_column = n_column_sub;
//        columns.n_row = n_row;
//        //columns.csc_val.resize(nnz_sub);
//        //columns.csc_row_idx.resize(nnz_sub);
//        //columns.csc_col_ptr.resize(n_column_sub + 1);
//        //csr data
//        columns.csr_val.resize(nnz_sub);
//        columns.csr_row_ptr.resize(dataset.csr_row_ptr.size());
//        columns.csr_col_idx.resize(nnz_sub);
//        
//        //csr copy
//        columns.csr_val.copy_from(val.device_data(),nnz_sub);
//        columns.csr_col_idx.copy_from(col_idx.device_data(),nnz_sub);
//        columns.csr_row_ptr.copy_from(row_ptr.device_data(),dataset.csr_row_ptr.size());
//
//
//        columns.csc_val_origin.resize(nnz_sub);
//        columns.csc_row_idx_origin.resize(nnz_sub);
//        columns.csc_col_ptr_origin.resize(n_column_sub + 1);
//
//        //columns.csc_val.copy_from(csc_val.host_data() + first_col_start,
//        //                          nnz_sub);
//        //columns.csc_row_idx.copy_from(csc_row_idx.host_data() + first_col_start,
//        //                              nnz_sub);
//        //columns.csc_col_ptr.copy_from(csc_col_ptr.host_data() + first_col_id,
//        //                              n_column_sub + 1);
//
//        //origin
//        columns.csc_val_origin.copy_from(csc_val.device_data() + first_col_start,
//                                  nnz_sub);
//        columns.csc_row_idx_origin.copy_from(csc_row_idx.device_data() + first_col_start,
//                                      nnz_sub);
//        columns.csc_col_ptr_origin.copy_from(csc_col_ptr.device_data() + first_col_id,
//                                      n_column_sub + 1);
//        
//        int *csc_col_ptr_2d_data = columns.csc_col_ptr_origin.device_data();
//        correct_start(csc_col_ptr_2d_data, first_col_start, n_column_sub);
//        //columns.csc_col_ptr_origin.copy_from(columns.csc_col_ptr.host_data(),n_column_sub + 1);
//        // correct segment start positions
//        //LOG(TRACE) << "sorting feature values (multi-device)";
//        //cub_seg_sort_by_key(columns.csc_val_origin, columns.csc_row_idx_origin,
//        //                    columns.csc_col_ptr_origin, false);
//    });
//    csc_val.resize(0);
//    csc_col_ptr.resize(0);
//    csc_row_idx.resize(0);
//    //set pointer?
//    val.resize(0);
//    col_idx.resize(0);
//    row_ptr.resize(0);
//    //consume much time when cub_seg_sort_by_key
//    SyncMem::clear_cache();
//    auto t_end = timer.now();
//    std::chrono::duration<float> used_time = t_end - t_start;
//    LOG(INFO) << "Converting csr to csc using time: " << used_time.count()
//              << " s";
//}

void SparseColumns::csr2csc_cpu(
    const DataSet &dataset, vector<std::unique_ptr<SparseColumns>> &v_columns) {
    LOG(INFO) << "convert csr to csc using cpu...";
    this->column_offset = 0;
    // cpu transpose
    n_column = dataset.n_features();
    n_row = dataset.n_instances();
    nnz = dataset.csr_val.size();

    float_type *csc_val_ptr = new float_type[nnz];
    int *csc_row_ptr = new int[nnz];
    int *csc_col_ptr = new int[n_column + 1];

    LOG(INFO) << string_format("#non-zeros = %ld, density = %.2f%%", nnz,
                               (float)nnz / n_column / dataset.n_instances() *
                                   100);
    for (int i = 0; i <= n_column; ++i) {
        csc_col_ptr[i] = 0;
    }

#pragma omp parallel for // about 5s
    for (int i = 0; i < nnz; ++i) {
        int idx = dataset.csr_col_idx[i] + 1;
#pragma omp atomic
        csc_col_ptr[idx] += 1;
    }

    //读数据有问题？？？？
    LOG(INFO)<<"16777217 should be 1? "<<csc_col_ptr[16777217];

    for (int i = 1; i < n_column + 1; ++i) {
        csc_col_ptr[i] += csc_col_ptr[i - 1];
    }

    // TODO to parallelize here
    for (int row = 0; row < dataset.n_instances(); ++row) {
        for (int j = dataset.csr_row_ptr[row]; j < dataset.csr_row_ptr[row + 1];
             ++j) {
            int col = dataset.csr_col_idx[j]; // csr col
            int dest = csc_col_ptr[col];      // destination index in csc array
            csc_val_ptr[dest] = dataset.csr_val[j];
            csc_row_ptr[dest] = row;
            csc_col_ptr[col] += 1; // increment sscolumn start position
        }
    }

    // recover column start position
    for (int i = 0, last = 0; i < n_column; ++i) {
        int next_last = csc_col_ptr[i];
        csc_col_ptr[i] = last;
        last = next_last;
    }
    
    //check correctness
    LOG(INFO)<<"20216829 feature have value? "<<csc_col_ptr[20216828]<<" "<<csc_col_ptr[20216829];
    //int t = 0;
    //for(int i=0;i<n_column + 1;++i){
    //    if(csc_col_ptr[i+1]-csc_col_ptr[i]>0){
    //        t++;
    //    }
    //}
    //LOG(INFO)<<"none zero feature num is "<<t;
    //save csc_col_ptr

    std::ofstream outfile("/home/xbr/ML_dataset/csc_col_kdda.txt");

    for(int i=0;i<n_column+1;i++){
        outfile << csc_col_ptr[i] << " ";
    }
    outfile.close();

    // split data to multiple device
    int n_device = v_columns.size();
    int ave_n_columns = n_column / n_device;
    DO_ON_MULTI_DEVICES(n_device, [&](int device_id) {
        SparseColumns &columns = *v_columns[device_id];
        int first_col_id = device_id * ave_n_columns;
        int n_column_sub = (device_id < n_device - 1) ? ave_n_columns
                                                      : n_column - first_col_id;
        int first_col_start = csc_col_ptr[first_col_id];
        int nnz_sub = (device_id < n_device - 1)
                          ? (csc_col_ptr[(device_id + 1) * ave_n_columns] -
                             first_col_start)
                          : (nnz - first_col_start);

        columns.column_offset = first_col_id + this->column_offset;
        columns.nnz = nnz_sub;
        columns.n_column = n_column_sub;
        columns.n_row = n_row;
        //columns.csc_val.resize(nnz_sub);
        //columns.csc_row_idx.resize(nnz_sub);
        //columns.csc_col_ptr.resize(n_column_sub + 1);

        //columns.csc_val.copy_from(csc_val_ptr + first_col_start, nnz_sub);
        //columns.csc_row_idx.copy_from(csc_row_ptr + first_col_start, nnz_sub);
        //columns.csc_col_ptr.copy_from(csc_col_ptr + first_col_id,
        //                              n_column_sub + 1);

        //int *csc_col_ptr_2d_data = columns.csc_col_ptr.host_data();
        //correct_start(csc_col_ptr_2d_data, first_col_start, n_column_sub);
        //seg_sort_by_key_cpu(columns.csc_val, columns.csc_row_idx,
        //                    columns.csc_col_ptr);
        columns.csc_val_origin.resize(nnz_sub);
        columns.csc_row_idx_origin.resize(nnz_sub);
        columns.csc_col_ptr_origin.resize(n_column_sub + 1);
        
        columns.csc_val_origin.copy_from(csc_val_ptr+first_col_start,nnz_sub);
        columns.csc_row_idx_origin.copy_from(csc_row_ptr + first_col_start,nnz_sub);
        columns.csc_col_ptr_origin.copy_from(csc_col_ptr + first_col_id,
                                      n_column_sub + 1);
        
        int *csc_col_ptr_2d_data = columns.csc_col_ptr_origin.device_data();
        correct_start(csc_col_ptr_2d_data, first_col_start, n_column_sub);
    });

    delete[](csc_val_ptr);
    delete[](csc_row_ptr);
    delete[](csc_col_ptr);
}

void SparseColumns::csc_by_default(
    const DataSet &dataset, vector<std::unique_ptr<SparseColumns>> &v_columns) {
    const float_type *csc_val_ptr = dataset.csc_val.data();
    const int *csc_row_ptr = dataset.csc_row_idx.data();
    const int *csc_col_ptr = dataset.csc_col_ptr.data();
    n_column = dataset.n_features();
    n_row = dataset.n_instances();
    nnz = dataset.csc_val.size();

    // split data to multiple device
    int n_device = v_columns.size();
    int ave_n_columns = n_column / n_device;
    DO_ON_MULTI_DEVICES(n_device, [&](int device_id) {
        SparseColumns &columns = *v_columns[device_id];
        int first_col_id = device_id * ave_n_columns;
        int n_column_sub = (device_id < n_device - 1) ? ave_n_columns
                                                      : n_column - first_col_id;
        int first_col_start = csc_col_ptr[first_col_id];
        int nnz_sub = (device_id < n_device - 1)
                          ? (csc_col_ptr[(device_id + 1) * ave_n_columns] -
                             first_col_start)
                          : (nnz - first_col_start);

        columns.column_offset = first_col_id + this->column_offset;
        columns.nnz = nnz_sub;

        columns.n_column = n_column_sub;
        columns.n_row = n_row;
        columns.csc_val.resize(nnz_sub);
        columns.csc_row_idx.resize(nnz_sub);
        columns.csc_col_ptr.resize(n_column_sub + 1);

        columns.csc_val.copy_from(csc_val_ptr + first_col_start, nnz_sub);
        columns.csc_row_idx.copy_from(csc_row_ptr + first_col_start, nnz_sub);
        columns.csc_col_ptr.copy_from(csc_col_ptr + first_col_id,
                                      n_column_sub + 1);

        int *csc_col_ptr_2d_data = columns.csc_col_ptr.host_data();
        correct_start(csc_col_ptr_2d_data, first_col_start, n_column_sub);
        cub_seg_sort_by_key(columns.csc_val, columns.csc_row_idx,
                            columns.csc_col_ptr, false);
    });
}


    

