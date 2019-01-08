#include "softmax.h"

namespace seq2seq{
    // cudnnSoftmaxAlgorithm_t: [CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_LOG]
    void Softmax_layer::init(cudnnSoftmaxAlgorithm_t alg) {
        _alg = alg;
        cudnn::createTensor4dDesc<float>(&_input_desc);
        cudnn::createTensor4dDesc<float>(&_output_desc);
    }

    void Softmax_layer::forward(Blob* input, Blob* output) {
        int batch = input->dim0;
        int num_labels = input->dim1;
        //    fprintf(stderr, "batch:%d num_labels:%d\n", batch, num_labels);

        int N = batch, K = num_labels, H = 1, W = 1;
        cudnn::setTensor4dDesc<float>(&_input_desc, N, K, H, W);
        cudnn::setTensor4dDesc<float>(&_output_desc, N, K, H, W);

        cudnnErrCheck(cudnnSoftmaxForward(GlobalAssets::instance()->cudnnHandle(),
                    _alg, CUDNN_SOFTMAX_MODE_INSTANCE,
                    cudnn::dataType<float>::one, _input_desc, input->device_w,
                    cudnn::dataType<float>::zero, _output_desc, output->device_w));
    }

    void Softmax_layer::backward(Blob* input, Blob* output) {
        cudnnErrCheck(cudnnSoftmaxBackward(GlobalAssets::instance()->cudnnHandle(),
                    _alg, CUDNN_SOFTMAX_MODE_INSTANCE,
                    cudnn::dataType<float>::one, _output_desc, output->device_w,
                    _output_desc, output->device_g,
                    cudnn::dataType<float>::zero, _input_desc, input->device_g));
    }
}
