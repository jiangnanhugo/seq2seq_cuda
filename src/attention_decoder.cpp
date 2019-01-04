#include "attention_decoder.h"

namespace seq2seq {
	void AttentionDecoder::init(int batch_size, int hidden_size, int input_size,
		int alignment_model_size /* = -1 */, int maxout_size /* = -1 */,
		int max_source_seq_len /* = 128 */, int max_target_seq_len /* = 128 */) {

		_batch_size = batch_size;
		_hidden_size = hidden_size;
		_input_size = input_size;
		_max_source_seq_len = max_source_seq_len;
		_max_target_seq_len = max_target_seq_len;

		// if not set, defaults it to hidden_size
		_alignment_model_size = alignment_model_size == -1 ? hidden_size : alignment_model_size;

		// if not set, defaults it to hidden_size
		_maxout_size = maxout_size == -1 ? hidden_size : maxout_size;

		fprintf(stderr, "batch_size: %d hidden_size: %d input_size: %d " \
			"max_source_seq_len: %d max_target_seq_len: %d alignment_model_size: %d maxout_size: %d\n",
			_batch_size, _hidden_size, _input_size, _max_source_seq_len, _max_target_seq_len, _alignment_model_size, _maxout_size);

		// param w
		_param_w.set_dim(input_size, 3, hidden_size);
		_param_w.malloced();
		// TODO: initialize rnn parameters using paper stated
		xavier_fill(_param_w.host_data, _param_w.size(), input_size, hidden_size);
		_param_w.copy_data_to_device();

		// param u
		_param_u.set_dim(hidden_size, 3, hidden_size);
		_param_u.malloced();
		// TODO: initialize rnn parameters using paper stated
		xavier_fill(_param_u.host_data, _param_u.size(), hidden_size, hidden_size);
		_param_u.copy_data_to_device();

		// param c
		_param_c.set_dim(2 * hidden_size, 3, hidden_size);
		_param_c.malloced();
		// TODO: initialize rnn parameters using paper stated
		xavier_fill(_param_c.host_data, _param_c.size(), 2 * hidden_size, hidden_size);
		_param_c.copy_data_to_device();


		// param attention w
		_param_att_w.set_dim(hidden_size, _alignment_model_size);
		_param_att_w.malloced();
		xavier_fill(_param_att_w.host_data, _param_att_w.size(), hidden_size, _alignment_model_size);
		_param_att_w.copy_data_to_device();

		// param attention u
		_param_att_u.set_dim(2 * hidden_size, _alignment_model_size);
		_param_att_u.malloced();
		xavier_fill(_param_att_u.host_data, _param_att_u.size(), 2 * hidden_size, _alignment_model_size);
		_param_att_u.copy_data_to_device();

		// param attention v
		_param_att_v.set_dim(_alignment_model_size, 1);
		_param_att_v.malloced();
		xavier_fill(_param_att_v.host_data, _param_att_v.size(), _alignment_model_size, 1);
		_param_att_v.copy_data_to_device();

		// param maxout u
		_param_m_u.set_dim(_hidden_size, 2 * _maxout_size);
		_param_m_u.malloced();
		xavier_fill(_param_m_u.host_data, _param_m_u.size(), _hidden_size, 2 * _maxout_size);
		_param_m_u.copy_data_to_device();

		// param maxout v
		_param_m_v.set_dim(_input_size, 2 * _maxout_size);
		_param_m_v.malloced();
		xavier_fill(_param_m_v.host_data, _param_m_v.size(), _input_size, 2 * _maxout_size);
		_param_m_v.copy_data_to_device();

		// param maxout c
		_param_m_c.set_dim(2 * _hidden_size, 2 * _maxout_size);
		_param_m_c.malloced();
		xavier_fill(_param_m_c.host_data, _param_m_c.size(), 2 * _hidden_size, 2 * _maxout_size);
		_param_m_c.copy_data_to_device();

		///////////////////////////////////////////////////
		// gates
		_pre_gate.set_dim(_max_target_seq_len, _batch_size, 9 * _hidden_size);
		_pre_gate.malloced();

		_gate.set_dim(_max_target_seq_len, _batch_size, 3 * _hidden_size);
		_gate.malloced();

		///////////////////////////////////////////////////////////////
		// alignment model related
		_context.set_dim(_max_target_seq_len, _batch_size, 2 * _hidden_size);
		_context.malloced();

		_attention_weights.set_dim(_max_target_seq_len, _max_source_seq_len, _batch_size);
		_attention_weights.malloced();

		_attention_scores.set_dim(_max_target_seq_len, _max_source_seq_len, _batch_size);
		_attention_scores.malloced();

		// it actually should be a 4-rank tensor
		_alignment_feats.set_dim(_max_target_seq_len * _max_source_seq_len, _batch_size, _alignment_model_size);
		_alignment_feats.malloced();

		_at_w_terms.set_dim(_max_target_seq_len, _batch_size, _alignment_model_size);
		_at_w_terms.malloced();

		// these terms can be pre computead (only related to encoder_hidden, and param_att_u)
		_at_u_terms.set_dim(_max_source_seq_len, _batch_size, _alignment_model_size);
		_at_u_terms.malloced();

		_softmax_alg = CUDNN_SOFTMAX_ACCURATE;
		cudnn::createTensor4dDesc<float>(&_softmax_input_desc);
		cudnn::createTensor4dDesc<float>(&_softmax_output_desc);

		///////////////////////////////////////////////////////////////
		// maxout related, decoder_hidden will be intermedia, maxout result will be returned
		_decoder_hidden.set_dim(_max_target_seq_len, _batch_size, _hidden_size);
		_decoder_hidden.malloced();

		_pre_maxout.set_dim(_max_target_seq_len, _batch_size, 2 * _maxout_size);
		_pre_maxout.malloced();

		_max_ele_idx.set_dim(_max_target_seq_len, _batch_size, _maxout_size);
		_max_ele_idx.malloced();

		///////////////////////////////////////////////////////////////
		// initial hidden
		_h0.set_dim(_batch_size, _hidden_size);
		_h0.malloced();
		initGPUData(_h0.device_data, _h0.size(), 0.0f);
	}

	void AttentionDecoder::const Blob* encoder_hidden, const float* h_data_tm1, const int t) {
		// at_w_terms = T.dot(h_data_tm1, At_W)
		gpu_gemm(CblasNoTrans, CblasNoTrans,
			_batch_size, _alignment_model_size, _hidden_size,
			1.0f, h_data_tm1, _param_att_w.device_data, 0.0f,
			_at_w_terms.device_data + t * _batch_size * _alignment_model_size);

		// added = T.tile(at_w_terms, (seq_length, 1)) + at_u_terms
		// e_t = T.tanh(added)
		add_at_w_and_u_terms_and_nonlinear(
			_at_w_terms.device_data + t * _batch_size * _alignment_model_size,
			_at_u_terms.device_data, _alignment_feats.device_data + t * _source_seq_len * _batch_size * _alignment_model_size,
			_source_seq_len, _batch_size, _alignment_model_size);

		// score_t = T.dot(e_t, At_V).reshape((seq_length, batch_size))
		gpu_gemm(CblasNoTrans, CblasNoTrans,
			_source_seq_len * _batch_size, 1, _alignment_model_size,
			1.0f, _alignment_feats.device_data + t * _source_seq_len * _batch_size * _alignment_model_size,
			_param_att_v.device_data, 0.0f,
			_attention_scores.device_data + t * _source_seq_len * _batch_size);

		// a_t = T.nnet.softmax(score_t.transpose()).transpose()
		int N = 1, K = _source_seq_len, H = _batch_size, W = 1;
		cudnn::setTensor4dDesc<float>(&_softmax_input_desc, N, K, H, W);
		cudnn::setTensor4dDesc<float>(&_softmax_output_desc, N, K, H, W);

		cudnnErrCheck(cudnnSoftmaxForward(
			GlobalAssets::instance()->cudnnHandle(),
			_softmax_alg,
			CUDNN_SOFTMAX_MODE_CHANNEL,
			cudnn::dataType<float>::one,
			_softmax_input_desc,
			_attention_scores.device_data + t * _source_seq_len * _batch_size,
			cudnn::dataType<float>::zero,
			_softmax_output_desc,
			_attention_weights.device_data + t * _source_seq_len * _batch_size));

		// a_t_repeated = a_t.repeat(2 * hidden_size).reshape((a_t.shape[0], a_t.shape[1], 2 * hidden_size))
		compute_context(_attention_weights.device_data + t * _source_seq_len * _batch_size,
			encoder_hidden->device_data,
			_context.device_data + t * _batch_size * 2 * _hidden_size,
			_source_seq_len, _batch_size, _hidden_size);
	}

	void AttentionDecoder::bp_dynamic_context(Blob* encoder_hidden, const float* h_data_tm1, float* h_diff_tm1, const int t) {
		bp_compute_context(_context.device_diff + t * _batch_size * 2 * _hidden_size,
			_attention_weights.device_data + t * _source_seq_len * _batch_size,
			encoder_hidden->device_data,
			_attention_weights.device_diff + t * _source_seq_len * _batch_size,
			encoder_hidden->device_diff,
			_source_seq_len, _batch_size, _hidden_size);

		// bp softmax
		cudnnErrCheck(cudnnSoftmaxBackward(
			GlobalAssets::instance()->cudnnHandle(),
			_softmax_alg,
			CUDNN_SOFTMAX_MODE_CHANNEL,
			cudnn::dataType<float>::one,
			_softmax_output_desc,
			_attention_weights.device_data + t * _source_seq_len * _batch_size,
			_softmax_output_desc,
			_attention_weights.device_diff + t * _source_seq_len * _batch_size,
			cudnn::dataType<float>::zero,
			_softmax_input_desc,
			_attention_scores.device_diff + t * _source_seq_len * _batch_size));

		// grads wrt _param_att_v
		gpu_gemm(CblasTrans, CblasNoTrans,
			_alignment_model_size, 1, _source_seq_len * _batch_size,
			1.0f, _alignment_feats.device_data + t * _source_seq_len * _batch_size * _alignment_model_size,
			_attention_scores.device_diff + t * _source_seq_len * _batch_size, 1.0f,
			_param_att_v.device_diff);

		// grads wrt _alignment_feats
		gpu_gemm(CblasNoTrans, CblasTrans,
			_source_seq_len * _batch_size, _alignment_model_size, 1,
            1.0f, _attention_scores.device_diff + t * _source_seq_len * _batch_size, _param_att_v.device_data, 1.0f,
			_alignment_feats.device_diff + t * _source_seq_len * _batch_size * _alignment_model_size);


		// TODO: bp to inside tanh, needs to write a kernel
		// added = T.tile(at_w_terms, (seq_length, 1)) + at_u_terms
		// e_t = T.tanh(added)
		add_at_w_and_u_terms_and_nonlinear_bp(
			_alignment_feats.device_data + t * _source_seq_len * _batch_size * _alignment_model_size,
			_alignment_feats.device_diff + t * _source_seq_len * _batch_size * _alignment_model_size,
			_at_w_terms.device_diff + t * _batch_size * _alignment_model_size,
			_at_u_terms.device_diff,
			_source_seq_len, _batch_size, _alignment_model_size);

		// grads wrt _param_att_w
		gpu_gemm(CblasTrans, CblasNoTrans,
			_hidden_size, _alignment_model_size, _batch_size,
			1.0f, h_data_tm1, _at_w_terms.device_diff + t * _batch_size * _alignment_model_size, 1.0f,
			_param_att_w.device_diff);

		// grads wrt h_tm1
		gpu_gemm(CblasNoTrans, CblasTrans,
			_batch_size, _hidden_size, _alignment_model_size,
			1.0f, _at_w_terms.device_diff + t * _batch_size * _alignment_model_size, _param_att_w.device_data, 1.0f,
			h_diff_tm1);
		// NOTICE: grads wrt encoder_hidden, and grads wrt _param_att_w, will be computed after recurrent bp ends
	}


    void AttentionDecoder::maxout(Blob* _pre_maxout, Blob* input, Blob* output){
    	float* pre_maxout_data = _pre_maxout.device_data;
    	gpu_gemm(CblasNoTrans, CblasNoTrans,
			_target_seq_len * _batch_size, 2 * _maxout_size, _hidden_size,
            1.0f, _decoder_hidden.device_data, _param_m_u.device_data, 0.0f,
    		pre_maxout_data);

    	gpu_gemm(CblasNoTrans, CblasNoTrans,
			_target_seq_len * _batch_size, 2 * _maxout_size, _input_size,
    		1.0f, input->device_data, _param_m_v.device_data, 1.0f,
    		pre_maxout_data);

    	gpu_gemm(CblasNoTrans, CblasNoTrans,
    		_target_seq_len * _batch_size, 2 * _maxout_size, 2 * _hidden_size,
			1.0f, _context.device_data, _param_m_c.device_data, 1.0f,
    		pre_maxout_data);

    	maxout_ff(pre_maxout_data, output->device_data, _max_ele_idx.device_data, _target_seq_len * _batch_size * _maxout_size);
    }

    void AttentionDecoder::pre_compute_data(Blob* input, Blob* encoder_hidden){
        _source_seq_len = encoder_hidden->dim0;
        _target_seq_len = input->dim0;
        // TODO: use a real h0 as initial (as stated in RNNSearch paper)
		this->compute_h0_ff(encoder_hidden);

        // compute dynamic context term
		// precompute _at_u_terms
		// at_u_terms = T.dot(reshaped_final_h, At_U)
		gpu_gemm(CblasNoTrans, CblasNoTrans,
			_source_seq_len * _batch_size, _alignment_model_size, 2 * _hidden_size,
			1.0f, encoder_hidden->device_data, _param_att_u.device_data, 0.0f,
			_at_u_terms.device_data);

        _pre_gate_data_w = _pre_gate.device_data;
		_pre_gate_data_u = _pre_gate.device_data + _target_seq_len * _batch_size * 3 * _hidden_size;
		_pre_gate_data_c = _pre_gate.device_data + _target_seq_len * _batch_size * 6 * _hidden_size;

		// precompute pregate of input to hidden
		gpu_gemm(CblasNoTrans, CblasNoTrans,
			_target_seq_len * _batch_size, 3 * _hidden_size, _input_size,
			1.0f, input->device_data, _param_w.device_data, 0.0f,
			_pre_gate_data_w);
    }

	void AttentionDecoder::recurrent(Blob* input, Blob* encoder_hidden, Blob* output) {
		_target_seq_len = input->dim0;
        this->compute_h0_ff(encoder_hidden);

        this->pre_compute_data(input, encoder_hidden);

		// N.B. : caution to order: reset(r) update(i) new gate(h)
		for (int t = 0; t < _target_seq_len; ++t) {
			// compute dynamic context
            this->step(encoder_hidden, t);
		}
		this->maxout(_pre_maxout, input, output);
	}

    void AttentionDecoder::step(Blob* encoder_hidden, int t){
        // compute dynamic context
        float* context_data_t = _context.device_data + t * _batch_size * 2 * _hidden_size;
        float* h_data_tm1 = t > 0 ? _decoder_hidden.device_data + (t - 1) * _batch_size * _hidden_size : _h0.device_data;

        float* h_data_t = _decoder_hidden.device_data + t * _batch_size * _hidden_size;
        float* pre_gate_data_w_t = _pre_gate_data_w + t * _batch_size * 3 * _hidden_size;
        float* pre_gate_data_u_t = _pre_gate_data_u + t * _batch_size * 3 * _hidden_size;
        float* pre_gate_data_c_t = _pre_gate_data_c + t * _batch_size * 3 * _hidden_size;
        float* gate_data_t     = _gate.device_data + t * _batch_size * 3 * _hidden_size;

        this->compute_dynamic_context(encoder_hidden, h_data_tm1, t);

        // compute pregate of hidden to hidden
        gpu_gemm(CblasNoTrans,CblasNoTrans,
            _batch_size, 3 * _hidden_size, _hidden_size,
            1.0f, h_data_tm1, _param_u.device_data, 0.0f,
            pre_gate_data_u_t);

        // compute pregate of context to hidden
        gpu_gemm(CblasNoTrans, CblasNoTrans,
            _batch_size, 3 * _hidden_size, 2 * _hidden_size,
            1.0f, context_data_t, _param_c.device_data, 0.0f,
            pre_gate_data_c_t);

        // for this time step, compute non linear and output
        attention_decoder_ff_nonlinear(h_data_tm1,
            pre_gate_data_w_t, pre_gate_data_u_t, pre_gate_data_c_t, gate_data_t, h_data_t,
            _batch_size, _hidden_size);
    }

    void AttentionDecoder::step(Blob* encoder_hidden, bool is_init){
        // compute dynamic context
        float* context_data_t = _context.device_data;
        float* h_data_tm1 = is_init == false ? _decoder_hidden.device_data : _h0.device_data;

        float* h_data_t = _decoder_hidden.device_data;
        float* pre_gate_data_w_t = _pre_gate_data_w;
        float* pre_gate_data_u_t = _pre_gate_data_u;
        float* pre_gate_data_c_t = _pre_gate_data_c;
        float* gate_data_t     = _gate.device_data;

        this->compute_dynamic_context(encoder_hidden, h_data_tm1, 0); // t = 0, _target_seq_len = 1

        // compute pregate of hidden to hidden
        gpu_gemm(CblasNoTrans, CblasNoTrans, _batch_size, 3 * _hidden_size, _hidden_size, 1.0f, h_data_tm1, _param_u.device_data, 0.0f, pre_gate_data_u_t);

        // compute pregate of context to hidden
        gpu_gemm(CblasNoTrans, CblasNoTrans, _batch_size, 3 * _hidden_size, 2 * _hidden_size, 1.0f, context_data_t, _param_c.device_data, 0.0f, pre_gate_data_c_t);

        // for this time step, compute non linear and output
        attention_decoder_ff_nonlinear(h_data_tm1,
            pre_gate_data_w_t, pre_gate_data_u_t, pre_gate_data_c_t, gate_data_t, h_data_t,
            _batch_size, _hidden_size);
    }

	void AttentionDecoder::set_all_diff_to_zero(Blob* input, Blob* encoder_hidden) {
		// memset diff to zero
		cudaErrCheck(cudaMemset(_h0.device_diff, 0.0, _h0.size() * sizeof(float)));
		cudaErrCheck(cudaMemset(input->device_diff, 0.0, input->size() * sizeof(float)));
		cudaErrCheck(cudaMemset(encoder_hidden->device_diff, 0.0, encoder_hidden->size() * sizeof(float)));

		cudaErrCheck(cudaMemset(_decoder_hidden.device_diff, 0.0, _decoder_hidden.size() * sizeof(float)));
		cudaErrCheck(cudaMemset(_pre_maxout.device_diff, 0.0, _pre_maxout.size() * sizeof(float)));

		cudaErrCheck(cudaMemset(_param_w.device_diff, 0.0, _param_w.size() * sizeof(float)));
		cudaErrCheck(cudaMemset(_param_u.device_diff, 0.0, _param_u.size() * sizeof(float)));
		cudaErrCheck(cudaMemset(_param_c.device_diff, 0.0, _param_c.size() * sizeof(float)));

		cudaErrCheck(cudaMemset(_param_att_w.device_diff, 0.0, _param_att_w.size() * sizeof(float)));
		cudaErrCheck(cudaMemset(_param_att_u.device_diff, 0.0, _param_att_u.size() * sizeof(float)));
		cudaErrCheck(cudaMemset(_param_att_v.device_diff, 0.0, _param_att_v.size() * sizeof(float)));

		cudaErrCheck(cudaMemset(_param_m_u.device_diff, 0.0, _param_m_u.size() * sizeof(float)));
		cudaErrCheck(cudaMemset(_param_m_v.device_diff, 0.0, _param_m_v.size() * sizeof(float)));
		cudaErrCheck(cudaMemset(_param_m_c.device_diff, 0.0, _param_m_c.size() * sizeof(float)));

		cudaErrCheck(cudaMemset(_pre_gate.device_diff, 0.0, _pre_gate.size() * sizeof(float)));
		cudaErrCheck(cudaMemset(_gate.device_diff, 0.0, _gate.size() * sizeof(float)));

		cudaErrCheck(cudaMemset(_context.device_diff, 0.0, _context.size() * sizeof(float)));
		cudaErrCheck(cudaMemset(_attention_weights.device_diff, 0.0, _attention_weights.size() * sizeof(float)));
		cudaErrCheck(cudaMemset(_attention_scores.device_diff, 0.0, _attention_scores.size() * sizeof(float)));
		cudaErrCheck(cudaMemset(_alignment_feats.device_diff, 0.0, _alignment_feats.size() * sizeof(float)));
		cudaErrCheck(cudaMemset(_at_w_terms.device_diff, 0.0, _at_w_terms.size() * sizeof(float)));
		cudaErrCheck(cudaMemset(_at_u_terms.device_diff, 0.0, _at_u_terms.size() * sizeof(float)));
	}


	void AttentionDecoder::backward(Blob* input, Blob* encoder_hidden, Blob* output) {
		_source_seq_len = encoder_hidden->dim0;
		_target_seq_len = input->dim0;

		this->set_all_diff_to_zero(input, encoder_hidden);

		////////////////////////////////////////////////////
			// maxout related back prop
		float* pre_maxout_diff = _pre_maxout.device_diff;
		float* pre_maxout_data = _pre_maxout.device_data;

		maxout_bp(pre_maxout_diff, output->device_diff, _max_ele_idx.device_data, _target_seq_len * _batch_size * _maxout_size);

#ifdef DEBUG_LOG
		fprintf(stderr, "pre_maxout_diff\n");
		_pre_maxout.copy_diff_to_host();
		display_matrix(_pre_maxout.host_diff, _target_seq_len, _batch_size, 2 * _maxout_size);
#endif


		// grads wrt param_m_u
		gpu_gemm(CblasTrans, CblasNoTrans,
			_hidden_size, 2 * _maxout_size, _target_seq_len * _batch_size,
			1.0f, _decoder_hidden.device_data, pre_maxout_diff, 0.0f,
			_param_m_u.device_diff);

		// grads wrt decoder_hidden
		gpu_gemm(CblasNoTrans, CblasTrans,
			_target_seq_len * _batch_size, _hidden_size, 2 * _maxout_size,
			1.0f, pre_maxout_diff, _param_m_u.device_data, 0.0f,
			_decoder_hidden.device_diff);


		// grads wrt param_m_v
		gpu_gemm(CblasTrans, CblasNoTrans,
			_input_size, 2 * _maxout_size, _target_seq_len * _batch_size,
			1.0f, input->device_data, pre_maxout_diff, 0.0f,
			_param_m_v.device_diff);

		// grads wrt input embedding
		gpu_gemm(CblasNoTrans, CblasTrans,
			_target_seq_len * _batch_size, _input_size, 2 * _maxout_size,
			1.0f, pre_maxout_diff, _param_m_v.device_data, 0.0f,
			input->device_diff);

		// grads wrt param_m_c
		gpu_gemm(CblasTrans, CblasNoTrans,
			2 * _hidden_size, 2 * _maxout_size, _target_seq_len * _batch_size,
			1.0f, _context.device_data, pre_maxout_diff, 0.0f,
			_param_m_c.device_diff);

		// grads wrt context
		gpu_gemm(CblasNoTrans, CblasTrans,
			_target_seq_len * _batch_size, 2 * _hidden_size, 2 * _maxout_size,
			1.0f, pre_maxout_diff, _param_m_c.device_data, 0.0f,
			_context.device_diff);

#ifdef DEBUG_LOG
		fprintf(stderr, "_context diff\n");
		_context.copy_diff_to_host();
		display_matrix(_context.host_diff, _target_seq_len, _batch_size, 2 * _hidden_size);

		fprintf(stderr, "_decoder_hidden diff\n");
		_decoder_hidden.copy_diff_to_host();
		display_matrix(_decoder_hidden.host_diff,
			_target_seq_len,
			_batch_size, _hidden_size);
#endif

		////////////////////////////////////////////////////
			// recurrent related back prop

		float* pre_gate_data_w = _pre_gate.device_data;
		float* pre_gate_data_u = _pre_gate.device_data \
			+ _target_seq_len * _batch_size * 3 * _hidden_size;
		float* pre_gate_data_c = _pre_gate.device_data \
			+ _target_seq_len * _batch_size * 6 * _hidden_size;

		float* pre_gate_diff_w = _pre_gate.device_diff;
		float* pre_gate_diff_u = _pre_gate.device_diff \
			+ _target_seq_len * _batch_size * 3 * _hidden_size;
		float* pre_gate_diff_c = _pre_gate.device_diff \
			+ _target_seq_len * _batch_size * 6 * _hidden_size;

		// back porp from last timestep to the first timestep
		for (int t = _target_seq_len - 1; t >= 0; --t) {
			// compute dynamic context
			float* h_data_tm1 = t > 0 ?
				_decoder_hidden.device_data + (t - 1) * _batch_size * _hidden_size :
				_h0.device_data;

			float* context_data_t = _context.device_data + t * _batch_size * 2 * _hidden_size;
			float* h_data_t = _decoder_hidden.device_data + t * _batch_size * _hidden_size;
			float* pre_gate_data_w_t = pre_gate_data_w + t * _batch_size * 3 * _hidden_size;
			float* pre_gate_data_u_t = pre_gate_data_u + t * _batch_size * 3 * _hidden_size;
			float* pre_gate_data_c_t = pre_gate_data_c + t * _batch_size * 3 * _hidden_size;
			float* gate_data_t = _gate.device_data + t * _batch_size * 3 * _hidden_size;

			float* h_diff_tm1 = t > 0 ?
				_decoder_hidden.device_diff + (t - 1) * _batch_size * _hidden_size :
				_h0.device_diff;

			float* context_diff_t = _context.device_diff + t * _batch_size * 2 * _hidden_size;
			float* h_diff_t = _decoder_hidden.device_diff + t * _batch_size * _hidden_size;
			float* pre_gate_diff_w_t = pre_gate_diff_w + t * _batch_size * 3 * _hidden_size;
			float* pre_gate_diff_u_t = pre_gate_diff_u + t * _batch_size * 3 * _hidden_size;
			float* pre_gate_diff_c_t = pre_gate_diff_c + t * _batch_size * 3 * _hidden_size;
			float* gate_diff_t = _gate.device_diff + t * _batch_size * 3 * _hidden_size;

			// for this time step, back prop non linear
			attention_decoder_bp_nonlinear(h_data_tm1, h_diff_t,
				gate_data_t, pre_gate_data_u_t, h_diff_tm1,
				pre_gate_diff_w_t, pre_gate_diff_u_t, pre_gate_diff_c_t, gate_diff_t,
				_batch_size, _hidden_size);


#ifdef DEBUG_LOG
			fprintf(stderr, "pre_gate_diff \n");
			_pre_gate.copy_diff_to_host();
			display_matrix(_pre_gate.host_diff, _pre_gate.dim0, _pre_gate.dim1, _pre_gate.dim2);

			fprintf(stderr, "gate_diff \n");
			_gate.copy_diff_to_host();
			display_matrix(_gate.host_diff, _gate.dim0, _gate.dim1, _gate.dim2);
#endif


			// grads wrt U
			gpu_gemm(CblasTrans, CblasNoTrans,
                _hidden_size, 3 * _hidden_size, _batch_size,
				1.0f,  h_data_tm1, pre_gate_diff_u_t, 1.0f,
				_param_u.device_diff);

			// grads wrt h_tm1
			gpu_gemm(CblasNoTrans, CblasTrans,
				_batch_size, _hidden_size, 3 * _hidden_size,
				1.0f, pre_gate_diff_u_t, _param_u.device_data, 1.0f,
				h_diff_tm1);

			// grads wrt C
			gpu_gemm(CblasTrans, CblasNoTrans,
				2 * _hidden_size, 3 * _hidden_size, _batch_size,
				1.0f, context_data_t, pre_gate_diff_c_t, 1.0f,
				_param_c.device_diff);

			// grads wrt c_i
			gpu_gemm(CblasNoTrans, CblasTrans,
				_batch_size, 2 * _hidden_size, 3 * _hidden_size,
				1.0f, pre_gate_diff_c_t, _param_c.device_data, 1.0f,
				context_diff_t);
			this->bp_dynamic_context(encoder_hidden, h_data_tm1, h_diff_tm1, t);
		}

		// one time computation for input projection
		// TODO: for w, no needs to memset to zero, and dont use adding in gemm call
		// grads wrt w
		gpu_gemm(CblasTrans, CblasNoTrans,
			_input_size, 3 * _hidden_size, _target_seq_len * _batch_size,
			1.0f, input->device_data, pre_gate_diff_w, 1.0f,
			_param_w.device_diff);

		// grads wrt input
		gpu_gemm(CblasNoTrans, CblasTrans,
			_target_seq_len * _batch_size, _input_size, 3 * _hidden_size,
			1.0f, pre_gate_diff_w, _param_w.device_data, 1.0f,
			input->device_diff);

		// grads wrt _param_att_u
		gpu_gemm(CblasTrans, CblasNoTrans,
			2 * _hidden_size, _alignment_model_size, _source_seq_len * _batch_size,
			1.0f, encoder_hidden->device_data, _at_u_terms.device_diff, 0.0f,
			_param_att_u.device_diff);

		// grads wrt encoder_hidden (add u terms part)
		gpu_gemm(CblasNoTrans, CblasTrans,
			_source_seq_len * _batch_size, 2 * _hidden_size, _alignment_model_size,
			1.0f, _at_u_terms.device_diff, _param_att_u.device_data, 1.0f,
			encoder_hidden->device_diff);

		this->compute_h0_bp(encoder_hidden);
	}

	// use encoder's reverse rnn's last words as decoder's h0
	// TODO: do a linear projection and non-linear as stated in paper
	void AttentionDecoder::compute_h0_ff(Blob* encoder_hidden) {
		copy_for_decoder_h0_data(encoder_hidden->device_data, _h0.device_data, _batch_size, _hidden_size);
#ifdef DEBUG_LOG
		fprintf(stderr, "_h0 data\n");
		_h0.copy_data_to_host();
		display_matrix(_h0.host_data, _batch_size, _hidden_size);
#endif
	}

	void AttentionDecoder::compute_h0_bp(Blob* encoder_hidden) {
#ifdef DEBUG_LOG
		fprintf(stderr, "_h0 diff\n");
		_h0.copy_diff_to_host();
		display_matrix(_h0.host_diff, _batch_size, _hidden_size);
#endif
		copy_for_decoder_h0_diff(_h0.device_diff, encoder_hidden->device_diff, _batch_size, _hidden_size);
	}

	void AttentionDecoder::display_all_params() {
		fprintf(stderr, "param_w \n");
		display_matrix(_param_w.host_data, _input_size, 3, _hidden_size);
		fprintf(stderr, "param_u \n");
		display_matrix(_param_u.host_data, _hidden_size, 3, _hidden_size);
		fprintf(stderr, "param_c \n");
		display_matrix(_param_c.host_data, 2 * _hidden_size, 3, _hidden_size);

		fprintf(stderr, "param_att_w %d %d\n", _hidden_size, _alignment_model_size);
		display_matrix(_param_att_w.host_data, _hidden_size, _alignment_model_size);
		fprintf(stderr, "param_att_u %d %d\n", 2 * _hidden_size, _alignment_model_size);
		display_matrix(_param_att_u.host_data, 2 * _hidden_size, _alignment_model_size);
		fprintf(stderr, "param_att_v %d %d\n", _alignment_model_size, 1);
		display_matrix(_param_att_v.host_data, _alignment_model_size, 1);

		fprintf(stderr, "param_m_u %d %d\n", _hidden_size, 2 * _maxout_size);
		display_matrix(_param_m_u.host_data, _hidden_size, 2 * _maxout_size);

		fprintf(stderr, "param_m_v %d %d\n", _input_size, 2 * _maxout_size);
		display_matrix(_param_m_v.host_data, _input_size, 2 * _maxout_size);

		fprintf(stderr, "param_m_c %d %d\n", 2 * _hidden_size, 2 * _maxout_size);
		display_matrix(_param_m_c.host_data, 2 * _hidden_size, 2 * _maxout_size);
	}

	void AttentionDecoder::display_all_params_diff() {
		_param_w.copy_diff_to_host();
		_param_u.copy_diff_to_host();
		_param_c.copy_diff_to_host();
		_param_att_w.copy_diff_to_host();
		_param_att_u.copy_diff_to_host();
		_param_att_v.copy_diff_to_host();
		_param_m_u.copy_diff_to_host();
		_param_m_v.copy_diff_to_host();
		_param_m_c.copy_diff_to_host();

		fprintf(stderr, "param_w diff\n");
		display_matrix(_param_w.host_diff, _input_size, 3, _hidden_size);
		fprintf(stderr, "param_u diff\n");
		display_matrix(_param_u.host_diff, _hidden_size, 3, _hidden_size);
		fprintf(stderr, "param_c diff\n");
		display_matrix(_param_c.host_diff, 2 * _hidden_size, 3, _hidden_size);

		fprintf(stderr, "param_att_w diff %d %d\n", _hidden_size, _alignment_model_size);
		display_matrix(_param_att_w.host_diff, _hidden_size, _alignment_model_size);
		fprintf(stderr, "param_att_u diff %d %d\n", 2 * _hidden_size, _alignment_model_size);
		display_matrix(_param_att_u.host_diff, 2 * _hidden_size, _alignment_model_size);
		fprintf(stderr, "param_att_v diff %d %d\n", _alignment_model_size, 1);
		display_matrix(_param_att_v.host_diff, _alignment_model_size, 1);

		fprintf(stderr, "param_m_u diff %d %d\n", _hidden_size, 2 * _maxout_size);
		display_matrix(_param_m_u.host_diff, _hidden_size, 2 * _maxout_size);

		fprintf(stderr, "param_m_v diff %d %d\n", _input_size, 2 * _maxout_size);
		display_matrix(_param_m_v.host_diff, _input_size, 2 * _maxout_size);

		fprintf(stderr, "param_m_c diff %d %d\n", 2 * _hidden_size, 2 * _maxout_size);
		display_matrix(_param_m_c.host_diff, 2 * _hidden_size, 2 * _maxout_size);
	}
}
