#ifndef SEQ2SEQ_INCLUDE_EMBD_H
#define SEQ2SEQ_INCLUDE_EMBD_H
#include "blob.h"
#include "gpu_common.h"

namespace seq2seq{
    class Emb_layer {
        public:
            void init(int voc_size, int emb_size);
            void forward(Blob* input, Blob* output);
            void backward(Blob* input, Blob* output);

            Blob* get_w() { return &_w;}
        private:
            int _voc_size, _emb_size;
            Blob _w;
    };

    void emb_ff(const float* w, const float* input, float* output, int seq_length, int batch_size, int emb_size);
    void emb_bp(float* w, const float* input, const float* grad_output, int seq_length, int batch_size, int emb_size);
}
#endif
