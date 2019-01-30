#include "rnn.h"

namespace seq2seq {
    void RNN_layer::init(int batch_size, int hidden_size, int input_size,
            bool is_train, int num_layers, bool bidirectional,
            cudnnRNNMode_t mode, float dropout) {

        _batch_size = batch_size;
        _hidden_size = hidden_size;
        _input_size = input_size;
        _is_train = is_train;
        _num_layers = num_layers;
        _bidirectional = bidirectional;
        _mode = mode;
        _dropout = dropout;

        // -------------------------
        // Set up inputs and outputs
        // -------------------------
        int dimA[3];
        int strideA[3];

        // In this example dimA[1] is constant across the whole sequence
        // This isn't required, all that is required is that it does not increase.
        for (unsigned int i = 0; i < max_allowd_seq_length; ++i) {
            cudnnErrCheck(cudnnCreateTensorDescriptor(&_x_desc[i]));
            cudnnErrCheck(cudnnCreateTensorDescriptor(&_y_desc[i]));
            cudnnErrCheck(cudnnCreateTensorDescriptor(&_dx_desc[i]));
            cudnnErrCheck(cudnnCreateTensorDescriptor(&_dy_desc[i]));

            dimA[0] = _batch_size; dimA[1] = _input_size; dimA[2] = 1;
            strideA[0] = dimA[1] * dimA[2]; strideA[1] = dimA[2]; strideA[2] = 1;

            cudnnErrCheck(cudnnSetTensorNdDescriptor(_x_desc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
            cudnnErrCheck(cudnnSetTensorNdDescriptor(_dx_desc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));

            dimA[0] = _batch_size; dimA[1] = _bidirectional ? _hidden_size * 2 : _hidden_size; dimA[2] = 1;

            strideA[0] = dimA[1] * dimA[2]; strideA[1] = dimA[2]; strideA[2] = 1;

            cudnnErrCheck(cudnnSetTensorNdDescriptor(_y_desc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
            cudnnErrCheck(cudnnSetTensorNdDescriptor(_dy_desc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
        }

        dimA[0] = _num_layers * (_bidirectional ? 2 : 1); dimA[1] = _batch_size; dimA[2] = _hidden_size;

        strideA[0] = dimA[1] * dimA[2]; strideA[1] = dimA[2]; strideA[2] = 1;

        cudnnErrCheck(cudnnCreateTensorDescriptor(&_hx_desc));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&_cx_desc));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&_hy_desc));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&_cy_desc));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&_dhx_desc));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&_dcx_desc));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&_dhy_desc));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&_dcy_desc));

        cudnnErrCheck(cudnnSetTensorNdDescriptor(_hx_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
        cudnnErrCheck(cudnnSetTensorNdDescriptor(_cx_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
        cudnnErrCheck(cudnnSetTensorNdDescriptor(_hy_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
        cudnnErrCheck(cudnnSetTensorNdDescriptor(_cy_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
        cudnnErrCheck(cudnnSetTensorNdDescriptor(_dhx_desc, CUDNN_DATA_FLOAT, 3, dimA,strideA));
        cudnnErrCheck(cudnnSetTensorNdDescriptor(_dcx_desc, CUDNN_DATA_FLOAT, 3, dimA,strideA));
        cudnnErrCheck(cudnnSetTensorNdDescriptor(_dhy_desc, CUDNN_DATA_FLOAT, 3, dimA,strideA));
        cudnnErrCheck(cudnnSetTensorNdDescriptor(_dcy_desc, CUDNN_DATA_FLOAT, 3, dimA,strideA));

        // Set up the dropout descriptor (needed for the RNN descriptor)
        static const unsigned long long seed = 1337ull; // Pick a seed.

        cudnnErrCheck(cudnnCreateDropoutDescriptor(&_dropout_desc));

        // How much memory does dropout need for states?
        // These states are used to generate random numbers internally
        // and should not be freed until the RNN descriptor is no longer used
        size_t state_size;
        cudnnErrCheck(cudnnDropoutGetStatesSize(GlobalAssets::instance()->cudnnHandle(), &state_size));

        _states.reset(new GpuMemPtr(state_size));

        cudnnErrCheck(cudnnSetDropoutDescriptor(_dropout_desc, GlobalAssets::instance()->cudnnHandle(), _dropout,
                    _states->get(), state_size, seed));

        // -------------------------
        // Set up the RNN descriptor
        // -------------------------
        cudnnErrCheck(cudnnCreateRNNDescriptor(&_rnn_desc));
        cudnnErrCheck(cudnnSetRNNDescriptor(
                    GlobalAssets::instance()->cudnnHandle(), _rnn_desc, _hidden_size, _num_layers,
                    _dropout_desc, // cudnnDropoutDescriptor_t dropoutDesc,
                    CUDNN_LINEAR_INPUT, // We can also skip the input matrix transformation
                    _bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, // cudnnDirectionMode_t direction
                    _mode,         // cudnnRNNMode_t mode
                    CUDNN_RNN_ALGO_STANDARD, // cudnnRNNAlgo_t algo
                    CUDNN_DATA_FLOAT)); // cudnnDataType dataType

        // -------------------------
        // Set up parameters
        // -------------------------
        // This needs to be done after the rnn descriptor is set as otherwise
        // we don't know how many parameters we have to allocate
        cudnnErrCheck(cudnnCreateFilterDescriptor(&_w_desc));
        cudnnErrCheck(cudnnCreateFilterDescriptor(&_dw_desc));

        cudnnErrCheck(cudnnGetRNNParamsSize(GlobalAssets::instance()->cudnnHandle(),
                    _rnn_desc, _x_desc[0], &_weights_size, CUDNN_DATA_FLOAT));

        //   fprintf(stderr, "rnn parameter size = %ld float numbers\n", _weights_size / sizeof(float));

        int dimW[3] = {_weights_size / sizeof(float), 1, 1};

        cudnnErrCheck(cudnnSetFilterNdDescriptor(_w_desc, CUDNN_DATA_FLOAT,
                    CUDNN_TENSOR_NCHW, 3, dimW));
        cudnnErrCheck(cudnnSetFilterNdDescriptor(_dw_desc, CUDNN_DATA_FLOAT,
                    CUDNN_TENSOR_NCHW, 3, dimW));

        _w.reset(new GpuMemPtr(_weights_size));
        _dw.reset(new GpuMemPtr(_weights_size));

        // -------------------------
        // Set up work space and reserved memory
        // -------------------------
        // Need for every pass
        cudnnErrCheck(cudnnGetRNNWorkspaceSize(
                    GlobalAssets::instance()->cudnnHandle(), _rnn_desc, max_allowd_seq_length,
                    _x_desc, &_work_size));

        // Only needed in training, shouldn't be touched between passes.
        cudnnErrCheck(cudnnGetRNNTrainingReserveSize(
                    GlobalAssets::instance()->cudnnHandle(), _rnn_desc, max_allowd_seq_length,
                    _x_desc, &_reserve_size));

        _work_space.reset(new GpuMemPtr(_work_size));
        _reserve_space.reset(new GpuMemPtr(_reserve_size));

        // Finally, initialize params
        this->initialize_params();
    }

    void RNN_layer::initialize_params() {
        // use queryed matrixes and biases
        int num_linear_layers = 0;
        if (_mode == CUDNN_RNN_RELU || _mode == CUDNN_RNN_TANH) {
            num_linear_layers = 2;
        } else if (_mode == CUDNN_LSTM) {
            num_linear_layers = 8;
        } else if (_mode == CUDNN_GRU) {
            num_linear_layers = 6;
        }

        int total_layers = _num_layers * (_bidirectional ? 2 : 1);
        _matrix_blobs.resize(total_layers);
        _bias_blobs.resize(total_layers);

        for (int layer = 0; layer < total_layers; ++layer) {
            // fprintf(stderr, "layer:%d\n", layer);
            _matrix_blobs[layer].resize(num_linear_layers);
            _bias_blobs[layer].resize(num_linear_layers);

            for (int lin_layer_id = 0; lin_layer_id < num_linear_layers; ++lin_layer_id) {
                // fprintf(stderr, "lin_layer_id:%d\n", lin_layer_id);
                Blob &matrix = _matrix_blobs[layer][lin_layer_id];
                Blob &bias = _bias_blobs[layer][lin_layer_id];

                cudnnFilterDescriptor_t linLayerMatDesc;
                cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerMatDesc));
                float *linLayerMat;

                cudnnErrCheck(cudnnGetRNNLinLayerMatrixParams(
                            GlobalAssets::instance()->cudnnHandle(), _rnn_desc, layer, _x_desc[0],
                            _w_desc, _w->get(), lin_layer_id, linLayerMatDesc, (void **)&linLayerMat));

                cudnnDataType_t dataType;
                cudnnTensorFormat_t format;
                int nbDims;
                int filterDimA[3];
                cudnnErrCheck(cudnnGetFilterNdDescriptor(linLayerMatDesc, 3, &dataType,
                            &format, &nbDims, filterDimA));

                // fprintf(stderr,"filterDimA: %d %d %d\n", filterDimA[0], filterDimA[1],
                // filterDimA[2]);
                initGPUData(linLayerMat, filterDimA[0] * filterDimA[1] * filterDimA[2], 0.1f);
                cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerMatDesc));

                matrix.device_w = static_cast<float *>(linLayerMat);
                // matrix.dim0 = filterDimA[0];
                matrix.dim0 = lin_layer_id < num_linear_layers / 2 ? _input_size : _hidden_size;
                matrix.dim1 = _hidden_size;

                // get bias
                cudnnFilterDescriptor_t linLayerBiasDesc;
                cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerBiasDesc));
                float *linLayerBias;

                cudnnErrCheck(cudnnGetRNNLinLayerBiasParams(
                            GlobalAssets::instance()->cudnnHandle(), _rnn_desc, layer, _x_desc[0],
                            _w_desc, _w->get(), lin_layer_id, linLayerBiasDesc,
                            (void **)&linLayerBias));

                cudnnErrCheck(cudnnGetFilterNdDescriptor(linLayerBiasDesc, 3, &dataType,
                            &format, &nbDims, filterDimA));

                // fprintf(stderr,"filterDimA: %d %d %d\n", filterDimA[0], filterDimA[1],
                // filterDimA[2]);
                initGPUData(linLayerBias, filterDimA[0] * filterDimA[1] * filterDimA[2], 0.1);

                cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerBiasDesc));

                bias.device_w = static_cast<float *>(linLayerBias);
                bias.dim0 = _hidden_size;
            }
        }

        // fill matrix
        for (size_t i = 0; i < _matrix_blobs.size(); ++i) {
            for (size_t j = 0; j < _matrix_blobs[i].size(); ++j) {
                Blob &matrix = _matrix_blobs[i][j];
                int shape =  matrix.size();
                matrix.malloced();
                xavier_fill(matrix.host_w, shape, matrix.dim0, matrix.dim1);
                matrix.copy_w_to_device();
            }
        }

        // fill bias
        for (size_t i = 0; i < _bias_blobs.size(); ++i) {
            for (size_t j = 0; j < _bias_blobs[i].size(); ++j) {
                Blob &bias = _bias_blobs[i][j];
                bias.malloced();
                constant_fill(bias.host_w, bias.size(), 0.0f);
                bias.copy_w_to_device();
            }
        }

        // use a whole blob for all of the parameters
        _param_blob.set_dim(_weights_size / sizeof(float), 1, 1);

        // TODO: be aware of this usage, especially when adding codes to free these memory
        _param_blob.device_w = static_cast<float *>(_w->get());
        _param_blob.device_g = static_cast<float *>(_dw->get());

        _param_blob.host_w = (float *)malloc(_param_blob.size() * sizeof(float));
        assert(_param_blob.host_w != NULL);
        _param_blob.host_g = (float *)malloc(_param_blob.size() * sizeof(float));
        assert(_param_blob.host_g != NULL);

        cudaErrCheck(cudaMalloc((void**)&_param_blob.device_m, _param_blob.size() * sizeof(float)));
        cudaErrCheck(cudaMemset(_param_blob.device_m, 0.0, _param_blob.size() * sizeof(float)));

        cudaErrCheck(cudaMalloc((void**)&_param_blob.device_v, _param_blob.size() * sizeof(float)));
        cudaErrCheck(cudaMemset(_param_blob.device_v, 0.0, _param_blob.size() * sizeof(float)));
    }

    // initial_hidden initial_cell = final_hidden final_cell = NULL
    void RNN_layer::forward(Blob *input, Blob *output, Blob *initial_hidden,
            Blob *initial_cell, Blob *final_hidden,
            Blob *final_cell) {
        int seq_length = input->dim0;
        int batch = input->dim1;
        int input_size = input->dim2;

        assert(seq_length <= max_allowd_seq_length); // length exceeded
        assert(batch == _batch_size); // batch_size not the same as stated in init
        assert(input_size == _input_size); // input_size not the same as stated in init
        if (_is_train) {
            cudnnErrCheck(cudnnRNNForwardTraining(
                        GlobalAssets::instance()->cudnnHandle(), _rnn_desc, seq_length, _x_desc,
                        input->device_w, _hx_desc,
                        initial_hidden == NULL ? NULL : initial_hidden->device_w, _cx_desc,
                        initial_cell == NULL ? NULL : initial_cell->device_w, _w_desc,
                        _w->get(), _y_desc, output->device_w, _hy_desc,
                        final_hidden == NULL ? NULL : final_hidden->device_w, _cy_desc,
                        final_cell == NULL ? NULL : final_cell->device_w, _work_space->get(),
                        _work_size, _reserve_space->get(), _reserve_size));
        } else {
            // printf("cudnnRNNForwardInference\n");
            cudnnErrCheck(cudnnRNNForwardInference(
                        GlobalAssets::instance()->cudnnHandle(), _rnn_desc, seq_length, _x_desc,
                        input->device_w, _hx_desc,
                        initial_hidden == NULL ? NULL : initial_hidden->device_w, _cx_desc,
                        initial_cell == NULL ? NULL : initial_cell->device_w, _w_desc,
                        _w->get(), _y_desc, output->device_w, _hy_desc,
                        final_hidden == NULL ? NULL : final_hidden->device_w, _cy_desc,
                        final_cell == NULL ? NULL : final_cell->device_w, _work_space->get(),
                        _work_size));
        }
    }

    // initial_hidden initial_cell = final_hidden final_cell = NULL
    void RNN_layer::backward(Blob *input, Blob *output, Blob *initial_hidden,
            Blob *initial_cell, Blob *final_hidden,
            Blob *final_cell) {

        int seq_length = input->dim0;
        int batch = input->dim1;
        int input_size = input->dim2;

        assert(seq_length <= max_allowd_seq_length); // length exceeded
        assert(batch == _batch_size); // batch_size not the same as stated in init
        assert(input_size == _input_size); // input_size not the same as stated in init

        // call backward
        cudnnErrCheck(cudnnRNNBackwardData(
                    GlobalAssets::instance()->cudnnHandle(), _rnn_desc, seq_length, _y_desc,
                    output->device_w, _dy_desc, output->device_g, _dhy_desc,
                    final_hidden == NULL ? NULL : final_hidden->device_g, _dcy_desc,
                    final_cell == NULL ? NULL : final_cell->device_g, _w_desc, _w->get(),
                    _hx_desc, initial_hidden == NULL ? NULL : initial_hidden->device_w,
                    _cx_desc, initial_cell == NULL ? NULL : initial_cell->device_w, _dx_desc,
                    input->device_g, _dhx_desc,
                    initial_hidden == NULL ? NULL : initial_hidden->device_g, _dcx_desc,
                    initial_cell == NULL ? NULL : initial_cell->device_g, _work_space->get(),
                    _work_size, _reserve_space->get(), _reserve_size));

        // cudnnRNNBackwardWeights adds to the data in dw.
        cudaErrCheck(cudaMemset(_dw->get(), 0, _weights_size));

        cudnnErrCheck(cudnnRNNBackwardWeights(
                    GlobalAssets::instance()->cudnnHandle(), _rnn_desc, seq_length, _x_desc,
                    input->device_w, _hx_desc,
                    initial_hidden == NULL ? NULL : initial_hidden->device_w, _y_desc,
                    output->device_w, _work_space->get(), _work_size, _dw_desc, _dw->get(),
                    _reserve_space->get(), _reserve_size));
    }
} // namespace seq2seq
