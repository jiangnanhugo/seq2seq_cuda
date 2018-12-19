#ifndef SEQ2SEQ_INCLUDE_SOFTMAX_H
#define SEQ2SEQ_INCLUDE_SOFTMAX_H

#include "blob.h"

namespace seq2seq{
    class Softmax_layer {
        public:
            // use CUDNN_SOFTMAX_ACCURATE for inference
            // use CUDNN_SOFTMAX_LOG for training (together with other loss parts)
            void init(cudnnSoftmaxAlgorithm_t alg = CUDNN_SOFTMAX_ACCURATE);
            void forward(Blob* input, Blob* output);
            void backward(Blob* input, Blob* output);
        private:
            cudnnTensorDescriptor_t _input_desc, _output_desc;
            cudnnSoftmaxAlgorithm_t _alg;
    };

    class TopKCompute{
        public:
            inline void init(int pad_id=0){ _pad_id=pad_id;}
            void forward(Blob *input, Blob* output, sort_type type, int ratio);

        private:
            int _pad_id;
    };
}
#endif
