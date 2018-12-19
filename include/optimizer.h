#ifndef SEQ2SEQ_INCLUDE_OPTIMIZER_H
#define SEQ2SEQ_INCLUDE_OPTIMIZER_H

#include "blob.h"
#include "gpu_common.h"

namespace seq2seq{
    enum OPTIMIZER_TYPE{SGD, SGDM, ADAM, RMSPROP, ADAGRAD, NESTROV};

    class Optimzer{
        public:
            inline void init(float lr, OPTIMIZER_TYPE optimizer_type){
                _lr = -lr;
                _optimizer_type = optimizer_type;
            }
            inline void set_lr(float lr){
                _lr=lr;
            }
            inline float get_lr(){
                return _lr;
            }
            void update(Blob* param);
            void Sgd(int size, float* diff, float *data);
            void Sgd_momentum(int size, float* moment, float* diff, float *data);

            // void Adam();
            // void RMSProp();
            // void Adagrad();
            // void Nestrov();

            OPTIMIZER_TYPE _optimizer_type;
            float _beta, _alpha, _lr;
    };
}
#endif
