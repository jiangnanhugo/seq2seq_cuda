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
    enum LOSS_TYPE{CROSS_ENTROPY, FOCAL_LOSS, OHEM};
    class Loss_layer {
        public:
            LOSS_TYPE _error_type;
            inline void init(int pad_id = 0, LOSS_TYPE type = CROSS_ENTROPY) {
                _pad_id = pad_id;
                _error_type =type;
            }
            // input:  num_labels * batch, labels: batch * 1, output: batch * 1 (loss values)
            void forward(Blob* input, Blob* labels, Blob* output);
            void backward(Blob* input, Blob* labels, Blob* output, float loss_factor);
        private:
            int _pad_id;
    };

    void OHEM_ff(const float *input, float *output, sort_type type, int size, float ratio, int pad_id);
    void TopK_ff(const float* input, float* output, int size, float K, int pad_id);

    class TopK_layer{
        public:
            inline void init(int pad_id=0){ _pad_id = pad_id;}
            void forward(float *input, float* output, int K);

        private:
            int _pad_id;
    };
}
#endif
