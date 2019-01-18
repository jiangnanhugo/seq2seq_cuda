#ifndef SEQ2SEQ_INCLUDE_BLOB_H
#define SEQ2SEQ_INCLUDE_BLOB_H
#include<string>
#include <cassert>
#include "common.h"
#include "cudnn_util.h"
using namespace std;

namespace seq2seq {
    // for simplicity, I use struct here
    struct Blob{
        int dim0, dim1, dim2;
        float *host_w, *device_w;      // param
        float *host_g, *device_g;      // gradient
        float *device_m, *device_v;      // momentum, velocity

        explicit Blob() : dim0(1), dim1(1), dim2(1) {}
        Blob(int d0, int d1, int d2){ dim0=d0; dim1=d1; dim2=d2;}
        int size() {return dim0 * dim1 * dim2;}
        void set_dim(int d0, int d1){ dim0=d0; dim1=d1; dim2=1;}
        void set_dim(int d0, int d1, int d2){dim0=d0; dim1=d1; dim2=d2;}

        void malloced();
        void copy_w_to_device();
        void copy_w_to_host();
        void copy_grad_to_host();
        // saving matrix (ignore dim3) into a text file
        void savetxt(const string& filename);
        // loading matrix (ignore dim3) into a text file
        void loadtxt(const string& filename);
        // for debug purpose
        void show_w(const string& info = "");
        void show_grad(const string& info = "");
    };

    // this can be protected by a shared_ptr
    class GpuMemPtr{
        public:
            explicit GpuMemPtr() : _data(NULL) {}

            explicit GpuMemPtr(size_t size_in_bytes) {
                cudaErrCheck(cudaMalloc((void**)&_data, size_in_bytes));
            }
            ~GpuMemPtr() {
                if (_data != NULL) {
                    cudaFree(_data);
                }
            }
            void* get() {return _data;}
        private:
            void* _data;
            DISALLOW_COPY_AND_ASSIGN(GpuMemPtr)
    };

}
#endif
