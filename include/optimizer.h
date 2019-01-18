#ifndef SEQ2SEQ_INCLUDE_OPTIMIZER_H
#define SEQ2SEQ_INCLUDE_OPTIMIZER_H

#include "blob.h"
#include "gpu_common.h"
#include <cmath>

namespace seq2seq{
    enum OPTIMIZER_TYPE{SGD, SGDM, ADAM, NESTROV};

    class Optimzer{
        public:
            inline void init(float lr, OPTIMIZER_TYPE optimizer_type){
                _lr = lr;
                _optimizer_type = optimizer_type;
            }
            inline void set_lr(float lr){
                _lr = lr;
            }
            inline float get_lr(){
                return _lr;
            }
            inline void set_t(float t){
                _t = t;
            }
            inline float get_t(){
                return _t;
            }
            void update(Blob* param);
            void Sgd(float *w, float* grad, int size);
            void Sgd_momentum(float *w, float* g, float* m, int size);
            void Nestrov(float *w, float *g, float *m, int size);
            void Adam(float *w, float* g, float* m, float* v, int size);

            OPTIMIZER_TYPE _optimizer_type;
            float _lr, _t;                      // learning_rate, time_step
    };

    void nestrov_update(float *w, float *g, float *m, int N, const float beta, const float lr);

    void adam_update(float* w, float* g, float* m, float * v,
                    int N, float beta1, float beta2, float correction, float eps, const float lr);
}
#endif
