#include "embd.h"

namespace seq2seq{
    void EmbCompute::init(int voc_size, int emb_size) {
		_voc_size = voc_size;
		_emb_size = emb_size;

		// prepare weights
		_w.set_dim(voc_size, emb_size);
		_w.malloc_data();
		xavier_fill(_w.host_data, _w.size(), voc_size, emb_size);
		_w.copy_data_to_device();
	}

    // input: dim0 * dim1 =  seq_length * batch_size
    // weights: dim0 * dim1 = voc_size * emb_size
    // output: seq_length * batch_size * emb_size
	void EmbCompute::forward(Blob* input, Blob* output) {
		int batch = input->dim0;
		emb_ff(_w.device_data, input->device_data, output->device_data, batch, input->dim1, _emb_size);
	}

	void EmbCompute::backward(Blob* input, Blob* output){
		int seq_length = input->dim0;
        int batch_size = input->dim1;
		emb_bp(_w.device_diff, input->device_data, output->device_diff, seq_length, batch_size, _emb_size);
	}



    // embedding ff/bp for feeding to rnn compute
    // ff result shape is seq_length * batch * emb_size
}
