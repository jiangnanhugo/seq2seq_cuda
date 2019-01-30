#include "embd.h"

namespace seq2seq{
    void Emb_layer::init(int voc_size, int emb_size) {
		_voc_size = voc_size;
		_emb_size = emb_size;

		// prepare weights
		_w.set_dim(voc_size, emb_size);
		_w.malloced();
		xavier_fill(_w.host_w, _w.size(), voc_size, emb_size);
		_w.copy_w_to_device();
	}

    // begin cuda_kernel
    __global__
    void emb_ff_kernel(const float* w, const float* input, float* output, int N, int emb_size){
        CUDA_KERNEL_LOOP(i, N) {
            int row = i / emb_size;
            int col = i % emb_size;
            const float* emb_t = w + static_cast<unsigned int>(input[row]) * emb_size;
            output[i] = emb_t[col];
        }
    }

    void emb_ff(const float* w, const float* input, float* output, int N, int emb_size){
        const dim3 blockSize(CUDA_NUM_THREADS, 1, 1);
        const dim3 gridSize(GET_BLOCKS(N), 1, 1);
        emb_ff_kernel<<< gridSize, blockSize >>> (w, input, output, N, emb_size);
    }
    // end cuda_kernel

    // input: dim0 * dim1 =  seq_length * batch_size
    // weights: dim0 * dim1 = voc_size * emb_size
    // output: seq_length * batch_size * emb_size
	void Emb_layer::forward(Blob* input, Blob* output) {
		int batch_size = input->dim0;
        int seq_len = input->dim1;
        int N = seq_len * batch_size * _emb_size;
        assert(output->size()==N);
		emb_ff(_w.device_w, input->device_w, output->device_w, N, _emb_size);
	}

    __global__
    void emb_bp_kernel(float* w, const float* input, const float* grad_output, int N, int emb_size) {
        CUDA_KERNEL_LOOP(i, N) {
            int row = i / emb_size;
            int col = i % emb_size;

            float* emb_t = w + static_cast<unsigned int>(input[row]) * emb_size;
            atomicAdd(emb_t + col, grad_output[i]);
        }
    }

    void emb_bp(float* w, const float* input, const float* grad_output, int N, int emb_size) {
        emb_bp_kernel<<<GET_BLOCKS(N), CUDA_NUM_THREADS>>>(w, input, grad_output, N, emb_size);
    }

	void Emb_layer::backward(Blob* input, Blob* output){
		int seq_len = input->dim0;
		int batch_size = input->dim1;
        int N = seq_len * batch_size * _emb_size;
        assert(output->size()==N);
		emb_bp(_w.device_g, input->device_w, output->device_g, N, _emb_size);
	}
}
