#include "activation.h"

namespace seq2seq {
	void ActivationCompute::init(cudnnActivationMode_t mode /* = CUDNN_ACTIVATION_SIGMOID */) {
		_mode = mode;
		cudnnErrCheck(cudnnCreateActivationDescriptor(&_activ_desc));
		cudnnErrCheck(cudnnSetActivationDescriptor(_activ_desc, _mode, CUDNN_PROPAGATE_NAN, 0.0));
		cudnn::createTensor4dDesc<float>(&_input_desc);
		cudnn::createTensor4dDesc<float>(&_output_desc);
	}

	void ActivationCompute::forward(Blob* input, Blob* output) {
		// input dim0 * dim1 = batch size * num
		int N = input->dim0;    // batch_size
		int K = input->dim1;    // num
		int H = 1;
		int W = 1;
		cudnn::setTensor4dDesc<float>(&_input_desc, N, K, H, W);
		cudnn::setTensor4dDesc<float>(&_output_desc, N, K, H, W);

		cudnnErrCheck(cudnnActivationForward(GlobalAssets::instance()->cudnnHandle(),
			_activ_desc,
			cudnn::dataType<float>::one,  _input_desc,  input->device_data,
			cudnn::dataType<float>::zero, _output_desc, output->device_data));
	}

	void ActivationCompute::backward(Blob* input, Blob* output) {
		cudnnErrCheck(cudnnActivationBackward(GlobalAssets::instance()->cudnnHandle(),
			_activ_desc, cudnn::dataType<float>::one, _output_desc, output->device_data, _output_desc,
			output->device_diff, _input_desc, input->device_data,
			cudnn::dataType<float>::zero, _input_desc, input->device_diff));
	}
} // namespace seq2seq
