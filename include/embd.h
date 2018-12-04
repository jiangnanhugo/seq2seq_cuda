#ifndef SEQ2SEQ_INCLUDE_EMBD_H
#define SEQ2SEQ_INCLUDE_EMBD_H
#include "blob.h"
#include "gpu_common.h"

namespace seq2seq{
    class EmbCompute {
        public:
            void init(int voc_size, int emb_size);
            void forward(Blob* input, Blob* output);
            void backward(Blob* input, Blob* output);

            Blob* get_w() { return &_w;}
        private:
            int _voc_size, _emb_size;
            Blob _w;
    };
}
#endif
