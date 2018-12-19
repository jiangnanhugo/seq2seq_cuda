#ifndef SEQ2SEQ_INCLUDE_LOSS_H
#define SEQ2SEQ_INCLUDE_LOSS_H

#include "blob.h"
#include "gpu_common.h"

namespace seq2seq{
    // brief: NegativeLossCompute assumes the input is the result
    // of SoftmaxCompute with CUDNN_SOFTMAX_LOG option
    // a pad symbol id is required to set all of the diff to zero
    // if that target is a pad symbol
    // i.e.: only implements "negative" in negative log loss
    enum LOSS_TYPE{CROSS_ENTROPY, FOCAL_LOSS};
    class Loss_layer {
        public:
            LOSS_TYPE _error_type;
            inline void init(int pad_id = 0, LOSS_TYPE type=CROSS_ENTROPY) {
                _pad_id = pad_id;
                _error_type =type;
            }
            // input shape:  num_labels * batch
            // labels shape: batch * 1
            // output shape: batch * 1 (loss values)
            void forward(Blob* input, Blob* labels, Blob* output);
            void backward(Blob* input, Blob* labels, Blob* output, float loss_factor);
        private:
            int _pad_id;
    };
}
#endif
