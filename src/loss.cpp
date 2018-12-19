#include"loss.h"

namespace seq2seq{
    void Loss_layer::forward(Blob* input, Blob* labels, Blob* output) {
		int batch = input->dim0;
		int num_labels = input->dim1;
        if(_error_type==LOSS_TYPE::CROSS_ENTROPY){
            cross_entropy_loss_ff(input->device_data, labels->device_data, output->device_data,
						          batch, num_labels, _pad_id);
        }else if(_error_type==LOSS_TYPE::FOCAL_LOSS){
            float gamma=2.0;
            focal_loss_ff(input->device_data, labels->device_data, output->device_data, batch, num_labels, gamma, _pad_id);
        }
	}

	// output is the loss values, dont need for now
	void Loss_layer::backward(Blob* input, Blob* labels, Blob* output, float loss_factor) {
		int batch = input->dim0;
		int num_labels = input->dim1;
        if(_error_type==LOSS_TYPE::CROSS_ENTROPY){
            cross_entropy_loss_bp(input->device_data, labels->device_data, input->device_diff, batch, num_labels, loss_factor,_pad_id);
        }else if(_error_type==LOSS_TYPE::FOCAL_LOSS){
            focal_loss_bp(input->device_data, labels->device_data, input->device_diff, batch, num_labels, loss_factor,_pad_id);
        }
	}
}
