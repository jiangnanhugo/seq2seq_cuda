#include "optimizer.h"

namespace seq2seq {

void Optimzer::update(Blob *param) {
  if (_optimizer_type == OPTIMIZER_TYPE::SGD) {
    Sgd(param->device_w, param->device_g, param->size());
  } else if (_optimizer_type == OPTIMIZER_TYPE::SGDM) {
    Sgd_momentum(param->device_w, param->device_g, param->device_m,
                 param->size());
  } else if (_optimizer_type == OPTIMIZER_TYPE::NESTROV) {
    Nestrov(param->device_w, param->device_g, param->device_m, param->size());
  } else if (_optimizer_type == OPTIMIZER_TYPE::ADAM) {
    Adam(param->device_w, param->device_g, param->device_m, param->device_v,
         param->size());
  }
}
void Optimzer::Sgd(float *w, float *grad, int size) {
  // w = _lr * grad + w
  cublasErrCheck(cublasSaxpy(GlobalAssets::instance()->cublasHandle(), size,
                             &_lr, grad, 1, w, 1));
}

void Optimzer::Sgd_momentum(float *w, float *g, float *m, int size) {
  // moment = beta * moment + grad
  const float beta = 0.9;
  cublasErrCheck(cublasSaxpy(GlobalAssets::instance()->cublasHandle(), size,
                             &beta, m, 1, g, 1));
  // w = _lr * moment + w
  cublasErrCheck(cublasSaxpy(GlobalAssets::instance()->cublasHandle(), size,
                             &_lr, m, 1, w, 1));
}

__global__
void nestrov_update_kernel(float *w, float *g, float *m, int N, const float beta,
                                      const float lr) {
  CUDA_KERNEL_LOOP(i, N) {
    const float mi = m[i];
    float mi_new = lr * g[i] + beta * m[i];
    float ng = (1 + beta) * mi_new + beta * mi;
    w[i] += ng;
  }
}

void nestrov_update(float *w, float *g, float *m, int N, const float beta, const float lr) {
  const dim3 blockSize(CUDA_NUM_THREADS, 1, 1);
  const dim3 gridSize(GET_BLOCKS(N), 1, 1);
  nestrov_update_kernel<<<gridSize, blockSize>>>(w, g, m, N, beta, lr);
}

void Optimzer::Nestrov(float *w, float *g, float *m, int size) {
  const float beta = 0.9;
  nestrov_update(w, g, m, size, beta, _lr);
}

__global__
void adam_update_kernel(float *w, float *g, float *m, float *v,
                                   int N, float beta1, float beta2,
                                   float correction, float eps,
                                   const float lr) {
  CUDA_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float mi = m[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = v[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    float ng = lr * correction * mi / (sqrtf(vi) + eps);
    w[i] += ng;
  }
}

void adam_update(float *w, float *g, float *m, float *v, int N, float beta1,
                 float beta2, float correction, float eps, const float lr) {
  const dim3 blockSize(CUDA_NUM_THREADS, 1, 1);
  const dim3 gridSize(GET_BLOCKS(N), 1, 1);
  adam_update_kernel<<<gridSize, blockSize>>>(w, g, m, v, N, beta1, beta2,
                                              correction, eps, lr);
}

void Optimzer::Adam(float *w, float *g, float *m, float *v, int size) {
  const float eps = 1e-8, beta1 = 0.9, beta2 = 0.999;
  const float correction = sqrt(1. - pow(beta2, _t)) / (1. - pow(beta1, _t));
  adam_update(w, g, m, v, size, beta1, beta2, correction, eps, _lr);
}

//
// void Optimzer::RMSProp(){
//
// }
// void Optimzer::Adagrad(){
//
// }
//
//

} // namespace seq2seq
