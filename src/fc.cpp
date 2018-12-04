#include "fc.h"

namespace seq2seq{
    void FCCompute::init(int input_size, int output_size) {
        _input_size=input_size;
        _output_size=output_size;

        // prepare weights
        _w.set_dim(_input_size, _output_size);
        _w.malloc_data();
        xavier_fill(_w.host_data, _w.size(), _input_size, _output_size);
        _w.copy_data_to_device();

        // prepare bias
        _b.set_dim(1, _output_size);
        _b.malloc_data();
        constant_fill(_b.host_data, _output_size, 0.0f);
        _b.copy_data_to_device();

        // prepare bias_multiplier
        _bias_multiplier.set_dim(max_allowd_batch, 1);
        _bias_multiplier.malloc_data();
        constant_fill(_bias_multiplier.host_data, max_allowd_batch, 1.0f);
        _bias_multiplier.copy_data_to_device();
    }

    void FCCompute::forward(Blob* input, Blob* output) {
        // input dim0 * dim1 = batch size * _input_sizeput
        // weights dim0 * dim1 = _input_sizeput * _output_size
        int batch = input->dim0;
        // weight
        gpu_gemm(CblasNoTrans, CblasNoTrans, batch, _output_size, _input_size, 1.0f, input->device_data, _w.device_data, 0.0f, output->device_data);
        // add bias
        gpu_gemm(CblasNoTrans, CblasNoTrans, batch, _output_size,1, 1.0f, _bias_multiplier.device_data, _b.device_data, 1.0f, output->device_data);
    }

    void FCCompute::backward(Blob* input, Blob* output) {
        int batch = input->dim0;
        // grads wrt w
        gpu_gemm(CblasTrans, CblasNoTrans, _input_size, _output_size, batch, 1.0f, input->device_data, output->device_diff, 0.0f, _w.device_diff);
        // grads wrt b
        gpu_gemv(CblasTrans, batch, _output_size, 1.0f, output->device_diff, _bias_multiplier.device_data, 0.0f, _b.device_diff);
        // grads wrt input
        gpu_gemm(CblasNoTrans, CblasTrans, batch, _input_size, _output_size, 1.0f, output->device_diff, _w.device_data, 0.0f, input->device_diff);
    }
}
