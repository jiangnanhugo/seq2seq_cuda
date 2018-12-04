#ifndef SEQ2SEQ_INCLUDE_MODEL_H
#define SEQ2SEQ_INCLUDE_MODEL_H

#include "data_reader.h"
#include "init.h"
#include "attention_decoder.h"

namespace seq2seq{
    struct Seq2SeqModel{
        void init(int max_encoder_seq_len, int max_decoder_seq_len, string loss_type, string optimizer_type, float lr);
        float forward(Blob* encoder_input, Blob* decoder_input, Blob* decoder_target);
        void backward(Blob* encoder_input, Blob* decoder_input, Blob* decoder_target);
        void clip_gradients(float max_gradient_norm);
        void optimize(Blob* encoder_input, Blob* decoder_input);
        void set_lr(float lr);

        void set_param(int source_voc_size, int target_voc_size, int batch_size, int emb_size, int hidden_size);
        void load_model(const string& dirname);
        void save_model(const string& dirname);

        EmbCompute encoder_emb, decoder_emb;
        RNNCompute encoder_rnn;
        AttentionDecoder decoder_rnn;
        FCCompute fc_compute;
        SoftmaxCompute softmax;
        LossCompute loss_compute;
        Optimzer optimizer;

        Blob encoder_emb_blob, decoder_emb_blob,
             encoder_rnn_blob, encoder_rnn_final_hidden, decoder_rnn_blob;
        Blob presoftmax_blob, softmax_result_blob, loss_blob;

        int _batch_size, _emb_size, _hidden_size, _alignment_size, _maxout_size;
        int _source_voc_size, _target_voc_size;
        vector<Blob *> _param_blobs;

        private:
        float _loss_factor;
    };
}
#endif
