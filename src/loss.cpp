#include <algorithm>
#include"loss.h"

namespace seq2seq{
    void Loss_layer::forward(Blob* input, Blob* labels, Blob* output) {
		int batch = input->dim0;
		int num_labels = input->dim1;
        if(_error_type == LOSS_TYPE::CROSS_ENTROPY){
            cross_entropy_loss_ff(input->device_w, labels->device_w, output->device_w,
						          batch, num_labels, _pad_id);
        }else if(_error_type == LOSS_TYPE::FOCAL_LOSS){
            float gamma=2.0;
            focal_loss_ff(input->device_w, labels->device_w, output->device_w, batch, num_labels, gamma, _pad_id);
        }
        // else if(_error_type == LOSS_TYPE::OHEM){
        //     sort_type type = sort_type::LARGE;
        //     float ratio = 0.5;
        //     OHEM_ff(input->device_w, labels->device_w, output->device_w, type, batch, ratio, _pad_id);
        // }
	}

	// output is the loss values, dont need for now
	void Loss_layer::backward(Blob* input, Blob* labels, Blob* output, float loss_factor) {
		int batch = input->dim0;
		int num_labels = input->dim1;
        if(_error_type == LOSS_TYPE::CROSS_ENTROPY){
            cross_entropy_loss_bp(input->device_w, labels->device_w, input->device_g, batch, num_labels, loss_factor,_pad_id);
        }else if(_error_type == LOSS_TYPE::FOCAL_LOSS){
            focal_loss_bp(input->device_w, labels->device_w, input->device_g, batch, num_labels, loss_factor,_pad_id);
        }
	}

    void OHEM_ff(const float *input, float *output, sort_type type, int size, float ratio, int pad_id){
        // sort the array
        for(int i = 0 ; i < size ; i++){
            output[i] = input[i];
        }
        std::sort(output, output + size);
        int len = size * (1. - ratio);
        if(type == sort_type::LARGE){
            for(int i = len; i < size; ++i){         // remove the bottom part
                output[i]=pad_id;
            }
        }else if (type == sort_type::SMALL){
            for(int i = 0 ; i < len ; ++i){             // remove the top part
                output[i] = 0;
            }
        }else if(type == sort_type::COMBO){
            for(int i = len/2 ; i < size - len/2 ; ++i){   // remove the middle part
                output[i]=pad_id;
            }
        }
    }
}
