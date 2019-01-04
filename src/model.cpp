#include "model.h"
#include <sys/stat.h>

namespace seq2seq{
    void Seq2SeqModel::init_train(int encoder_seq_len, int decoder_seq_len, string loss_type, string optimizer_type, float lr){
        // init layers
        encoder_emb_layer.init(_source_voc_size, _emb_size);
        decoder_emb_layer.init(_target_voc_size, _emb_size);
        encoder_rnn_layer.init(_batch_size, _hidden_size, _emb_size, true, 1, true, CUDNN_GRU, 0.0);
        decoder_rnn_layer.init(_batch_size, _hidden_size, _emb_size, _alignment_size, _maxout_size, encoder_seq_len, decoder_seq_len);
        linear_layer.init(_maxout_size, _target_voc_size);
        softmax_layer.init(CUDNN_SOFTMAX_LOG);

        if(loss_type.compare("cross_entropy")==0){
            loss_layer.init(DataReader::PAD_ID, LOSS_TYPE::CROSS_ENTROPY);
        }else if(loss_type.compare("focal_loss")==0){
            loss_layer.init(DataReader::PAD_ID, LOSS_TYPE::FOCAL_LOSS);
        }

        if(optimizer_type.compare("sgd")==0){
            optimizer.init(lr, OPTIMIZER_TYPE::SGD);
        }else if(optimizer_type.compare("sgd_m")==0){
            optimizer.init(lr, OPTIMIZER_TYPE::SGDM);
        }else if(optimizer_type.compare("adam")==0){
            optimizer.init(lr, OPTIMIZER_TYPE::ADAM);
        }else if(optimizer_type.compare("adagrad")==0){
            optimizer.init(lr, OPTIMIZER_TYPE::ADAGRAD);
        }else if(optimizer_type.compare("rmsprop")==0){
            optimizer.init(lr, OPTIMIZER_TYPE::RMSPROP);
        }else if(optimizer_type.compare("nestrov")==0){
            optimizer.init(lr, OPTIMIZER_TYPE::NESTROV);
        }

        // init intermedia blobs
        encoder_emb_blob.set_dim(encoder_seq_len, _batch_size, _emb_size);
        encoder_emb_blob.malloced();

        decoder_emb_blob.set_dim(decoder_seq_len, _batch_size, _emb_size);
        decoder_emb_blob.malloced();

        encoder_rnn_blob.set_dim(encoder_seq_len, _batch_size, 2 * _hidden_size);
        encoder_rnn_blob.malloced();

        encoder_rnn_final_hidden.set_dim(_batch_size, 2 * _hidden_size);
        encoder_rnn_final_hidden.malloced();

        decoder_rnn_blob.set_dim(decoder_seq_len, _batch_size, _hidden_size);
        decoder_rnn_blob.malloced();

        presoftmax_blob.set_dim(decoder_seq_len * _batch_size, _target_voc_size);
        presoftmax_blob.malloced();

        softmax_result_blob.set_dim(decoder_seq_len * _batch_size, _target_voc_size);
        softmax_result_blob.malloced();

        loss_blob.set_dim(decoder_seq_len * _batch_size, 1);
        loss_blob.malloced();

        _param_blobs.push_back(linear_layer.get_w());
        _param_blobs.push_back(linear_layer.get_b());
        _param_blobs.push_back(decoder_rnn_layer.param_w());
        _param_blobs.push_back(decoder_rnn_layer.param_u());
        _param_blobs.push_back(decoder_rnn_layer.param_c());
        _param_blobs.push_back(decoder_rnn_layer.param_att_w());
        _param_blobs.push_back(decoder_rnn_layer.param_att_u());
        _param_blobs.push_back(decoder_rnn_layer.param_att_v());
        _param_blobs.push_back(decoder_rnn_layer.param_m_u());
        _param_blobs.push_back(decoder_rnn_layer.param_m_v());
        _param_blobs.push_back(decoder_rnn_layer.param_m_c());
        _param_blobs.push_back(encoder_rnn_layer.get_param());
        _param_blobs.push_back(decoder_emb_layer.get_w());
        _param_blobs.push_back(encoder_emb_layer.get_w());
    }

    void Seq2SeqModel::init_inference(int encoder_seq_len, int beam_size){
        // init layers
        encoder_emb_layer.init(_source_voc_size, _emb_size);
        decoder_emb_layer.init(_target_voc_size, _emb_size);
        encoder_rnn_layer.init(_batch_size, _hidden_size, _emb_size, true, 1, true, CUDNN_GRU, 0.0);
        decoder_rnn_layer.init(_batch_size, _hidden_size, _emb_size, _alignment_size, _maxout_size, encoder_seq_len, 1);
        linear_layer.init(_maxout_size, _target_voc_size);
        softmax_layer.init(CUDNN_SOFTMAX_LOG);

        // init intermedia blobs
        // _batch_size = 1
        encoder_emb_blob.set_dim(encoder_seq_len, 1, _emb_size);
        encoder_emb_blob.malloced();

        decoder_emb_blob.set_dim(beam_size, 1, _emb_size);
        decoder_emb_blob.malloced();

        encoder_rnn_blob.set_dim(encoder_seq_len, 1, 2 * _hidden_size);
        encoder_rnn_blob.malloced();

        encoder_rnn_final_hidden.set_dim(1, 2 * _hidden_size);
        encoder_rnn_final_hidden.malloced();

        decoder_rnn_blob.set_dim(1 , beam_size, _hidden_size);
        decoder_rnn_blob.malloced();

        presoftmax_blob.set_dim(beam_size, _target_voc_size);
        presoftmax_blob.malloced();

        softmax_result_blob.set_dim(beam_size, _target_voc_size);
        softmax_result_blob.malloced();
    }

    void Seq2SeqModel::encode(Blob* encoder_input){
        // seq_len * batch * emb_size
        encoder_emb_blob.set_dim(encoder_input->dim0, encoder_input->dim1, _emb_size);
        encoder_emb_layer.forward(encoder_input, &encoder_emb_blob);

        // seq_len * batch * 2 * hidden_size
        encoder_rnn_blob.set_dim(encoder_emb_blob.dim0, encoder_emb_blob.dim1, 2 * _hidden_size);

        // batch * 2 * hidden_size
        encoder_rnn_final_hidden.set_dim(encoder_emb_blob.dim1, 2 * _hidden_size, 1);
        encoder_rnn_layer.forward(&encoder_emb_blob, &encoder_rnn_blob, NULL, &encoder_rnn_final_hidden, NULL);
    }

    float Seq2SeqModel::forward(Blob *encoder_input, Blob *decoder_input, Blob *decoder_target){
        this->encode(encoder_input);

        // seq_len * batch * emb_size
        decoder_emb_blob.set_dim(decoder_input->dim0, decoder_input->dim1, _emb_size);
        decoder_emb_layer.forward(decoder_input, &decoder_emb_blob);

        // seq_len * batch * hidden_size
        decoder_rnn_blob.set_dim(decoder_emb_blob.dim0, decoder_emb_blob.dim1, _hidden_size);
        decoder_rnn_layer.recurrent(&decoder_emb_blob, &encoder_rnn_blob, &decoder_rnn_blob);

        // before fc, reshpae the input to (seq_len * batch) * hidden_size
        decoder_rnn_blob.set_dim(decoder_rnn_blob.dim0 * decoder_rnn_blob.dim1, decoder_rnn_blob.dim2, 1);

        // result shape: (seq_len * batch) * target_voc_size
        presoftmax_blob.set_dim(decoder_rnn_blob.dim0, _target_voc_size);
        linear_layer.forward(&decoder_rnn_blob, &presoftmax_blob);

        // shape: (seq_len * batch) * target_voc_size
        softmax_result_blob.set_dim(presoftmax_blob.dim0, presoftmax_blob.dim1);
        softmax_layer.forward(&presoftmax_blob, &softmax_result_blob);

        // shape: (seq_len * batch) * 1
        loss_blob.set_dim(softmax_result_blob.dim0, 1);
        loss_layer.forward(&softmax_result_blob, decoder_target, &loss_blob);

        float total_loss = 0.0;
        int cnt = 0;

        loss_blob.copy_data_to_host();
        const float *losses = loss_blob.host_data;
        for (int i = 0; i < loss_blob.size(); ++i){
            if (fabs(losses[i]) > 1e-12){
                ++cnt;
                total_loss += losses[i];
            }
        }

        float avg_loss = 0.0;
        _loss_factor = 0.0;

        if (cnt > 0){
            avg_loss = total_loss / cnt;
            _loss_factor = 1.0 / cnt;
        }
        return avg_loss;
    }

    void Seq2SeqModel::step(Blob* decoder_input, bool is_init){
        // seq_len * batch * emb_size
        decoder_emb_blob.set_dim(1, decoder_input->dim1, _emb_size);
        decoder_emb_layer.forward(decoder_input, &decoder_emb_blob);

        // seq_len(=1) * batch * hidden_size
        decoder_rnn_blob.set_dim(1, decoder_emb_blob.dim1, _hidden_size);
        decoder_rnn_layer.pre_compute_data(&decoder_emb_blob, &encoder_rnn_blob);
        decoder_rnn_layer.step(&encoder_rnn_blob, is_init);
        decoder_rnn_layer.maxout(&decoder_emb_blob, &decoder_rnn_blob);

        // before fc, reshpae the input to [seq_len(=1) * batch, hidden_size]
        decoder_rnn_blob.set_dim(decoder_rnn_blob.dim1, decoder_rnn_blob.dim2, 1);

        // result shape: [seq_len(=1) * batch, target_voc_size]
        presoftmax_blob.set_dim(decoder_rnn_blob.dim0, _target_voc_size);
        linear_layer.forward(&decoder_rnn_blob, &presoftmax_blob);

        // shape: seq_len(=1) * batch, * target_voc_size
        softmax_result_blob.set_dim(presoftmax_blob.dim0, presoftmax_blob.dim1);
        softmax_layer.forward(&presoftmax_blob, &softmax_result_blob);

        softmax_result_blob.copy_data_to_host();
        float* probs = softmax_result_blob.host_data;
        int len =  _batch_size * _target_voc_size;
        decoder_input->copy_data_to_host();
        argsort(probs, decoder_input->host_data, _batch_size * _target_voc_size, _batch_size);
        decoder_input->copy_data_to_device();
    }

    void Seq2SeqModel::backward(Blob *encoder_input, Blob *decoder_input, Blob *decoder_target){
        loss_layer.backward(&softmax_result_blob, decoder_target, &loss_blob, _loss_factor);
        softmax_layer.backward(&presoftmax_blob, &softmax_result_blob);
        linear_layer.backward(&decoder_rnn_blob, &presoftmax_blob);
        decoder_rnn_blob.set_dim(decoder_emb_blob.dim0, decoder_emb_blob.dim1, _hidden_size);
        // attention to this call
        decoder_rnn_layer.backward(&decoder_emb_blob, &encoder_rnn_blob, &decoder_rnn_blob);
        // attention to this call
        encoder_rnn_layer.backward(&encoder_emb_blob, &encoder_rnn_blob, NULL, NULL, NULL);
        decoder_emb_layer.backward(decoder_input, &decoder_emb_blob);
        encoder_emb_layer.backward(encoder_input, &encoder_emb_blob);
    }

    void Seq2SeqModel::optimize(Blob *encoder_input, Blob *decoder_input){
        for (size_t i = 0; i < _param_blobs.size(); ++i){
            optimizer.update(_param_blobs[i]);
        }
    }

    void Seq2SeqModel::clip_gradients(float max_gradient_norm){
        float fc_sumsq = 0.0;
        float encoder_rnn_sumsq = 0.0;
        float decoder_rnn_sumsq = 0.0;

        cublasErrCheck(cublasSdot(GlobalAssets::instance()->cublasHandle(),
                    linear_layer.get_w()->size(), linear_layer.get_w()->device_diff, 1,
                    linear_layer.get_w()->device_diff, 1, &fc_sumsq));

        cublasErrCheck(cublasSdot(GlobalAssets::instance()->cublasHandle(),
                    encoder_rnn_layer.weights_size() / sizeof(float), encoder_rnn_layer.get_dw(), 1,
                    encoder_rnn_layer.get_dw(), 1, &encoder_rnn_sumsq));

        std::vector<Blob *> params;
        params.push_back(decoder_rnn_layer.param_w());
        params.push_back(decoder_rnn_layer.param_u());
        params.push_back(decoder_rnn_layer.param_c());
        params.push_back(decoder_rnn_layer.param_att_w());
        params.push_back(decoder_rnn_layer.param_att_u());
        params.push_back(decoder_rnn_layer.param_att_v());
        params.push_back(decoder_rnn_layer.param_m_u());
        params.push_back(decoder_rnn_layer.param_m_v());
        params.push_back(decoder_rnn_layer.param_m_c());

        for (size_t i = 0; i < params.size(); ++i){
            float temp_sumsq = 0.0;
            cublasErrCheck(cublasSdot(GlobalAssets::instance()->cublasHandle(), params[i]->size(), params[i]->device_diff, 1,
                        params[i]->device_diff, 1, &temp_sumsq));
            decoder_rnn_sumsq += temp_sumsq;
        }

        float gnorm = sqrt(fc_sumsq + encoder_rnn_sumsq + decoder_rnn_sumsq);          // global_norm

        if (gnorm > max_gradient_norm){
            fprintf(stderr, "global norm %.6f > thresh %.6f\n", gnorm, max_gradient_norm);

            float scale_factor = max_gradient_norm / gnorm;

            cublasErrCheck(cublasSscal(GlobalAssets::instance()->cublasHandle(),
                linear_layer.get_w()->size(), &scale_factor, linear_layer.get_w()->device_diff,1));

            cublasErrCheck(cublasSscal(GlobalAssets::instance()->cublasHandle(),
                        encoder_rnn_layer.weights_size() / sizeof(float),
                        &scale_factor, encoder_rnn_layer.get_dw(), 1));

            for (size_t i = 0; i < params.size(); ++i){
                cublasErrCheck(cublasSscal(GlobalAssets::instance()->cublasHandle(), params[i]->size(),
                            &scale_factor, params[i]->device_diff,1));
            }

            // althoug not count in embedding parameters when calculating global norm
            // also needs to scale embedding grads
            cublasErrCheck(cublasSscal(GlobalAssets::instance()->cublasHandle(), encoder_emb_blob.size(), &scale_factor, encoder_emb_blob.device_diff, 1));
            cublasErrCheck(cublasSscal(GlobalAssets::instance()->cublasHandle(), decoder_emb_blob.size(), &scale_factor, decoder_emb_blob.device_diff, 1));
        }
    }

    void Seq2SeqModel::set_param(int source_voc_size, int target_voc_size, int batch_size, int emb_size, int hidden_size){
        _source_voc_size = source_voc_size;
        _target_voc_size = target_voc_size;
        _batch_size = batch_size;
        _emb_size = emb_size;
        _hidden_size = hidden_size;
        _alignment_size = hidden_size;
        _maxout_size = hidden_size;
    }

    void Seq2SeqModel::set_lr_decay(float decay){
        float lr=optimizer.get_lr()*decay;
        optimizer.set_lr(lr);
    }

    // load it into text format
    void Seq2SeqModel::load_model(const string &dirname){
        encoder_emb_layer.get_w()->loadtxt(dirname + "/encoder.emb");
        decoder_emb_layer.get_w()->loadtxt(dirname + "/decoder.emb");

        encoder_rnn_layer.get_param()->loadtxt(dirname + "/encoder_rnn.weights");

        decoder_rnn_layer.param_w()->loadtxt(dirname + "/decoder_rnn.weights.w");
        decoder_rnn_layer.param_u()->loadtxt( dirname + "/decoder_rnn.weights.u");
        decoder_rnn_layer.param_c()->loadtxt(dirname + "/decoder_rnn.weights.c");
        decoder_rnn_layer.param_att_w()->loadtxt(dirname + "/decoder_rnn.weights.att_w");
        decoder_rnn_layer.param_att_u()->loadtxt(dirname + "/decoder_rnn.weights.att_u");
        decoder_rnn_layer.param_att_v()->loadtxt(dirname + "/decoder_rnn.weights.att_v");
        decoder_rnn_layer.param_m_u()->loadtxt(dirname + "/decoder_rnn.weights.m_u");
        decoder_rnn_layer.param_m_v()->loadtxt(dirname + "/decoder_rnn.weights.m_v");
        decoder_rnn_layer.param_m_c()->loadtxt(dirname + "/decoder_rnn.weights.m_c");

        linear_layer.get_w()->loadtxt(dirname + "/fc.weights");
        linear_layer.get_b()->loadtxt(dirname + "/fc.bias");
    }
    // save it into text format
    void Seq2SeqModel::save_model(const string &dirname){
        mkdir(dirname.c_str(), 0777);
        fprintf(stderr, "saving model to %s\n", dirname.c_str());
        encoder_emb_layer.get_w()->savetxt(dirname + "/encoder.emb");
        decoder_emb_layer.get_w()->savetxt(dirname + "/decoder.emb");

        encoder_rnn_layer.get_param()->savetxt(dirname + "/encoder_rnn.weights");
        decoder_rnn_layer.param_w()->savetxt(dirname + "/decoder_rnn.weights.w");
        decoder_rnn_layer.param_u()->savetxt(dirname + "/decoder_rnn.weights.u");
        decoder_rnn_layer.param_c()->savetxt(dirname + "/decoder_rnn.weights.c");

        decoder_rnn_layer.param_att_w()->savetxt(dirname + "/decoder_rnn.weights.att_w");
        decoder_rnn_layer.param_att_u()->savetxt(dirname + "/decoder_rnn.weights.att_u");
        decoder_rnn_layer.param_att_v()->savetxt(dirname + "/decoder_rnn.weights.att_v");

        decoder_rnn_layer.param_m_u()->savetxt(dirname + "/decoder_rnn.weights.m_u");
        decoder_rnn_layer.param_m_v()->savetxt(dirname + "/decoder_rnn.weights.m_v");
        decoder_rnn_layer.param_m_c()->savetxt(dirname + "/decoder_rnn.weights.m_c");

        linear_layer.get_w()->savetxt(dirname + "/fc.weights");
        linear_layer.get_b()->savetxt(dirname + "/fc.bias");
    }
} // namespace seq2seq
