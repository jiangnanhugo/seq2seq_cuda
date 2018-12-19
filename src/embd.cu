#include "embd.h"

namespace seq2seq{
    void Emb_layer::init(int voc_size, int emb_size) {
		_voc_size = voc_size;
		_emb_size = emb_size;

		// prepare weights
		_w.set_dim(voc_size, emb_size);
		_w.malloced();
		xavier_fill(_w.host_data, _w.size(), voc_size, emb_size);
		_w.copy_data_to_device();
	}

    // begin cuda_kernel
    __global__
    void emb_ff_kernel(const float* w, const float* input, float* output, int batch_size, int seq_length, int emb_size){
        int total = seq_length * batch_size * emb_size;
        CUDA_KERNEL_LOOP(i, total) {
            int row = i / emb_size;
            int column = i % emb_size;

            const float* emb_t = w + static_cast<unsigned int>(input[row]) * emb_size;
            output[i] = emb_t[column];
        }
    }

    void emb_ff(const float* w, const float* input, float* output, int batch_size, int seq_length, int emb_size){
        int total = seq_length * batch_size * emb_size;
        const dim3 blockSize(CUDA_NUM_THREADS, 1, 1);
        const dim3 gridSize(GET_BLOCKS(total), 1, 1);
        emb_ff_kernel<<< gridSize, blockSize >>> (w, input, output, batch_size, seq_length, emb_size);
    }
    // end cuda_kernel

    // input: dim0 * dim1 =  seq_length * batch_size
    // weights: dim0 * dim1 = voc_size * emb_size
    // output: seq_length * batch_size * emb_size
	void Emb_layer::forward(Blob* input, Blob* output) {
		int batch = input->dim0;
		emb_ff(_w.device_data, input->device_data, output->device_data, batch, input->dim1, _emb_size);
	}



    // begin emb layer

    __global__
    void emb_bp_kernel(float* w, const float* input, const float* grad_output, int seq_length, int batch_size, int emb_size) {
        int total = seq_length * batch_size * emb_size;
        CUDA_KERNEL_LOOP(i, total) {
            int row = i / emb_size;
            int column = i % emb_size;

            float* emb_t = w + static_cast<unsigned int>(input[row]) * emb_size;
            atomicAdd(emb_t + column, grad_output[i]);
        }
    }

    void emb_bp(float* w, const float* input, const float* grad_output, int seq_length, int batch_size, int emb_size) {
        int total = seq_length * batch_size * emb_size;
        emb_bp_kernel<<<GET_BLOCKS(total), CUDA_NUM_THREADS>>>(w, input, grad_output, seq_length, batch_size, emb_size);
    }



	void Emb_layer::backward(Blob* input, Blob* output){
		int seq_length = input->dim0;
        int batch_size = input->dim1;
		emb_bp(_w.device_diff, input->device_data, output->device_diff, seq_length, batch_size, _emb_size);
	}

    // embedding ff/bp for feeding to rnn compute
    // ff result shape is seq_length * batch * emb_size
}
