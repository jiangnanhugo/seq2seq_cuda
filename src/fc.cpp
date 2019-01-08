#include "fc.h"

namespace seq2seq{
    void Linear_layer::init(int input_size, int output_size) {
        _input_size=input_size;
        _output_size=output_size;

        // prepare weights
        _w.set_dim(_input_size, _output_size);
        _w.malloced();
        xavier_fill(_w.host_w, _w.size(), _input_size, _output_size);
        _w.copy_w_to_device();

        // prepare bias
        _b.set_dim(1, _output_size);
        _b.malloced();
        constant_fill(_b.host_w, _output_size, 0.0f);
        _b.copy_w_to_device();

        // prepare bias_multiplier
        _bias_multiplier.set_dim(max_allowd_batch, 1);
        _bias_multiplier.malloced();
        constant_fill(_bias_multiplier.host_w, max_allowd_batch, 1.0f);
        _bias_multiplier.copy_w_to_device();
    }

    void Linear_layer::forward(Blob* input, Blob* output) {
        // input dim0 * dim1 = batch size * _input_sizeput
        // weights dim0 * dim1 = _input_sizeput * _output_size
        int batch = input->dim0;
        // weight
        gpu_gemm(CblasNoTrans, CblasNoTrans, batch, _output_size, _input_size, 1.0f, input->device_w, _w.device_w, 0.0f, output->device_w);
        // add bias
        gpu_gemm(CblasNoTrans, CblasNoTrans, batch, _output_size, 1, 1.0f, _bias_multiplier.device_w, _b.device_w, 1.0f, output->device_w);
    }

    void Linear_layer::backward(Blob* input, Blob* output) {
        int batch = input->dim0;
        // grads wrt w
        gpu_gemm(CblasTrans, CblasNoTrans, _input_size, _output_size, batch, 1.0f, input->device_w, output->device_g, 0.0f, _w.device_g);
        // grads wrt b
        gpu_gemv(CblasTrans, batch, _output_size, 1.0f, output->device_g, _bias_multiplier.device_w, 0.0f, _b.device_g);
        // grads wrt input
        gpu_gemm(CblasNoTrans, CblasTrans, batch, _input_size, _output_size, 1.0f, output->device_g, _w.device_w, 0.0f, input->device_g);
    }
}
