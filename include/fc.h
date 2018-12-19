#ifndef SEQ2SEQ_INCLUDE_FC_H
#define SEQ2SEQ_INCLUDE_FC_H
#include "blob.h"

namespace seq2seq{
    class Linear_layer {
        public:
            void init(int input_size, int output_size);
            void forward(Blob* input, Blob* output);
            void backward(Blob* input, Blob* output);
            Blob* get_w() {return &_w;}
            Blob* get_b() {return &_b;}
        private:
            int _input_size, _output_size;
            Blob _w, _b, _bias_multiplier; // bias multiplier: broadcast vector to matrix.
            const int max_allowd_batch = 40960;
    };
}
#endif
