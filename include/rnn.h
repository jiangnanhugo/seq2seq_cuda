#ifndef SEQ2SEQ_INCLUDE_RNN_H
#define SEQ2SEQ_INCLUDE_RNN_H

#include "blob.h"

namespace seq2seq{
    class RNN_layer {
        public:
            void init(int batch_size, int hidden_size, int input_size,
                    bool is_training = true, int num_layers = 1, bool bidirectional = false,
                    cudnnRNNMode_t mode = CUDNN_GRU, float dropout = 0.0);

            /*
             * @brief forward pass of recurrent compute
             *
             * @param [in] input given x in shape: seq_length * batch * input_size
             * @param [out] output generate y in shape: seq_length * batch * hidden_size
             * @param [in] initial_hidden: in shape batch * input_size
             * @param [in] initial_cell: in shape batch * input_size
             * @param [out] final_hidden: in shape batch * hidden_size
             * @param [out] final_cell: in shape batch * hidden_size
             */
            void forward(Blob* input, Blob* output, Blob* initial_hidden = NULL, Blob* initial_cell = NULL,  Blob* final_hidden = NULL, Blob* final_cell = NULL);

            void backward(Blob* input, Blob* output,  Blob* initial_hidden = NULL, Blob* initial_cell = NULL, Blob* final_hidden = NULL, Blob* final_cell = NULL);

            inline Blob* get_param(){return &_param_blob;}
            inline vector<vector<Blob> >& get_matrix_blobs(){return _matrix_blobs;}
            inline vector<vector<Blob> >& get_bias_blobs(){return _bias_blobs;}
            inline float* get_dw() {return static_cast<float*>(_dw->get());}
            inline size_t weights_size() {return _weights_size;}

        private:
            // we will need this value to prepare descriptors for each time step
            static const int max_allowd_seq_length = 128;
            int _batch_size, _hidden_size, _input_size, _num_layers;
            bool _is_train, _bidirectional;
            cudnnRNNMode_t _mode;
            float _dropout;

            size_t _weights_size;
            cudnnFilterDescriptor_t _w_desc;        // all parameters descriptor
            cudnnFilterDescriptor_t _dw_desc;       // gradient of all parameters tensor descriptor
            shared_ptr<GpuMemPtr> _w;          // memory for all parameters
            shared_ptr<GpuMemPtr> _dw;         // memory for gradient of all parameters

            cudnnDropoutDescriptor_t _dropout_desc;
            shared_ptr<GpuMemPtr> _states; // memory for dropout internal

            size_t _work_size, _reserve_size;
            shared_ptr<GpuMemPtr> _work_space; // memory for dropout internal
            shared_ptr<GpuMemPtr> _reserve_space; // memory for dropout internal

            cudnnRNNDescriptor_t _rnn_desc;

            cudnnTensorDescriptor_t _x_desc[max_allowd_seq_length];    // input tensor descriptors
            cudnnTensorDescriptor_t _y_desc[max_allowd_seq_length];    // hidden tensor descriptors
            cudnnTensorDescriptor_t _dx_desc[max_allowd_seq_length];   // gradient of input tensor descriptors
            cudnnTensorDescriptor_t _dy_desc[max_allowd_seq_length];   // gradient of hidden tensor descriptors

            cudnnTensorDescriptor_t _hx_desc;             // initial hidden tensor descriptor
            cudnnTensorDescriptor_t _cx_desc;             // initial cell tensor descriptor
            cudnnTensorDescriptor_t _hy_desc;             // final hidden tensor descriptor
            cudnnTensorDescriptor_t _cy_desc;             // final cell tensor descriptor

            cudnnTensorDescriptor_t _dhx_desc;             // gradient of initial hidden tensor descriptor
            cudnnTensorDescriptor_t _dcx_desc;             // gradient of initial cell tensor descriptor
            cudnnTensorDescriptor_t _dhy_desc;             // gradient of final hidden tensor descriptor
            cudnnTensorDescriptor_t _dcy_desc;             // gradient of final cell tensor descriptor
        private:
            void initialize_params();
            // N.B. I will not use malloc_data of blob to manage the memory of param
            // instead, using codes in initialize_params to create/refer params
            Blob _param_blob;

            vector<vector<Blob> > _matrix_blobs, _bias_blobs;
    };
}
#endif
