#ifndef SEQ2SEQ_INCLUDE_MODEL_H
#define SEQ2SEQ_INCLUDE_MODEL_H

#include "data_reader.h"
#include "init.h"
#include "attention_decoder.h"

namespace seq2seq{
    struct Seq2SeqModel{
        void init_train(int encoder_seq_len, int decoder_seq_len, string loss_type, string optimizer_type, float lr);
        void init_inference(int encoder_seq_len, int beam_size, bool is_train);
        float forward(Blob* encoder_input, Blob* decoder_input, Blob* decoder_target);
        void backward(Blob* encoder_input, Blob* decoder_input, Blob* decoder_target);
        void encode(Blob* encoder_input);
        void step(Blob* decoder_input, int* parent_idx, int* word_idx, int beam_size, bool is_init);
        void clip_gradients(float max_gradient_norm);
        void optimize(Blob* encoder_input, Blob* decoder_input);
        void set_lr_decay(float decay);
        void inc_timestep();

        void set_param(int source_voc_size, int target_voc_size, int batch_size, int emb_size, int hidden_size);
        void load_model(const string& dirname);
        void save_model(const string& dirname);

        Emb_layer encoder_emb_layer, decoder_emb_layer;
        RNN_layer encoder_rnn_layer;
        AttentionDecoder decoder_rnn_layer;
        Linear_layer linear_layer;
        Softmax_layer softmax_layer;
        Loss_layer loss_layer;
        Optimzer optimizer;

        Blob encoder_emb_blob, decoder_emb_blob,
             encoder_rnn_blob, encoder_rnn_final_hidden, decoder_rnn_blob;
        Blob presoftmax_blob, softmax_result_blob, loss_blob;

        int _batch_size, _emb_size, _hidden_size, _alignment_size, _maxout_size;
        int _source_voc_size, _target_voc_size;
        vector<Blob *> _param_blobs;

        private:
        float _loss_factor;
        float _timestep;
    };
}
#endif
