#ifndef SEQ2SEQ_INCLUDE_ACTIVATION_H
#define SEQ2SEQ_INCLUDE_ACTIVATION_H

#include "blob.h"

namespace seq2seq {
    class Activation_function {
        public:
            // CUDNN_ACTIVATION_SIGMOID(0), CUDNN_ACTIVATION_RELU(1),
            // CUDNN_ACTIVATION_TANH(2), CUDNN_ACTIVATION_CLIPPED_RELU(3)
            void init(cudnnActivationMode_t mode = CUDNN_ACTIVATION_SIGMOID);
            void forward(Blob *input, Blob *output);
            void backward(Blob *input, Blob *output);
        private:
            cudnnTensorDescriptor_t _input_desc, _output_desc;
            cudnnActivationMode_t _mode;
            cudnnActivationDescriptor_t  _activ_desc;
    };
}

#endif
