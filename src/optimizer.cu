#include "optimizer.h"

namespace seq2seq{

    void Optimzer::update(Blob* param){
        if(_optimizer_type==OPTIMIZER_TYPE::SGD){
            Sgd(param->size(), param->device_diff, param->device_data);
        }else if(_optimizer_type==OPTIMIZER_TYPE::SGDM){
            Sgd_momentum(param->size(), param->device_moment1, param->device_diff, param->device_data);
        // }else if(_optimizer_type==OPTIMIZER_TYPE::ADAM){
        //     Adam();
        // }else if(_optimizer_type==OPTIMIZER_TYPE::RMSPROP){
        //     RMSProp();
        // }else if(_optimizer_type==OPTIMIZER_TYPE::ADAGRAD){
        //     Adagrad();
        // }else if(_optimizer_type==OPTIMIZER_TYPE::NESTROV){
        //     Nestrov();
        }
    }
    void Optimzer::Sgd(int size, float* diff, float *data){
        // data = _lr * diff + data
        cublasErrCheck(cublasSaxpy(GlobalAssets::instance()->cublasHandle(), size, &_lr, diff, 1, data, 1));
    }

    void Optimzer::Sgd_momentum(int size, float* moment, float* diff, float *data){
        // moment= beta * moment + diff
        cublasErrCheck(cublasSaxpy(GlobalAssets::instance()->cublasHandle(), size, &_beta, moment, 1, diff, 1));
        // data = _lr * moment + data
        cublasErrCheck(cublasSaxpy(GlobalAssets::instance()->cublasHandle(), size, &_lr, moment, 1, data, 1));
    }
    //
    // void Optimzer::Adam(){
    //
    // }
    //
    // __global__
    // void adam_kernel(const float* w, const float* input, float* output, int batch_size, int seq_length, int emb_size) {
    //     int total = seq_length * batch_size * emb_size;
    //     CUDA_KERNEL_LOOP(i, total) {
    //         int row = i / emb_size;
    //         int column = i % emb_size;
    //
    //         const float* emb_t = w + static_cast<unsigned int>(input[row]) * emb_size;
    //         output[i] = emb_t[column];
    //     }
    // }
    //
    // void Optimizer::adam(const float* w, const float* input, float* output, int batch_size, int seq_length, int emb_size) {
    //     int total = seq_length * batch_size * emb_size;
    //     const dim3 blockSize(CUDA_NUM_THREADS, 1, 1);
    //     const dim3 gridSize(GET_BLOCKS(total), 1, 1);
    //     adam_kernel<<< gridSize, blockSize >>> (w, input, output, batch_size, seq_length, emb_size);
    // }
    //
    // void Optimzer::RMSProp(){
    //
    // }
    // void Optimzer::Adagrad(){
    //
    // }
    //
    //
    // void Optimzer::Nestrov(){
    //
    // }


}
