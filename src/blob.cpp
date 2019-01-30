#include "blob.h"
#include <iostream>

namespace seq2seq{
    void Blob::malloced() {
        int shape = size();

        host_w = (float*)malloc(shape * sizeof(float));
        assert(host_w != NULL);
        cudaErrCheck(cudaMalloc((void**)&device_w, shape * sizeof(float)));
        cudaErrCheck(cudaMemset(device_w, 0.0, shape * sizeof(float)));

        host_g = (float*)malloc(shape * sizeof(float));
        assert(host_g != NULL);
        cudaErrCheck(cudaMalloc((void**)&device_g, shape * sizeof(float)));
        cudaErrCheck(cudaMemset(device_g, 0.0, shape * sizeof(float)));

        cudaErrCheck(cudaMalloc((void**)&device_m, shape * sizeof(float)));
        cudaErrCheck(cudaMemset(device_m, 0.0, shape * sizeof(float)));

        cudaErrCheck(cudaMalloc((void**)&device_v, shape * sizeof(float)));
        cudaErrCheck(cudaMemset(device_v, 0.0, shape * sizeof(float)));
    }

    void Blob::copy_w_to_device() {
        assert(host_w != NULL);
        assert(device_w != NULL);
        cudaErrCheck(cudaMemcpy(device_w, host_w, size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    void Blob::copy_w_to_host() {
        assert(host_w != NULL);
        assert(device_w != NULL);
        cudaErrCheck(cudaMemcpy(host_w, device_w, size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void Blob::copy_grad_to_host() {
        cudaErrCheck(cudaMemcpy(host_g, device_g, size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // saving matrix (ignore dim3) into a text file
    void Blob::savetxt(const string& filename) {
        this->copy_w_to_host();
        const float* data = this->host_w;
        FILE* fp = fopen(filename.c_str(), "w");
        int row = dim0, col = dim1 * dim2;
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < col; ++j) {
                fprintf(fp, "%+.8f%s", data[i * col + j], j == col - 1 ? "" : " ");
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }

    // loading matrix (ignore dim3) into a text file
    void Blob::loadtxt(const string& filename){
        float* data = this->host_w;
        ifstream infile(filename);
        assert(infile.good());

        string line;
        vector<string> strs;
        int row = dim0, col = dim1 * dim2;
        for (int i = 0; i < row; ++i) {
            getline(infile, line);
            split(line, strs);
            for (int j = 0; j < col; ++j) {
                data[i * col + j] = atof(strs[j].c_str());
            }
        }
        this->copy_w_to_device();
    }

    // for debug purpose
    void Blob::show_w(const string& info){
        this->copy_w_to_host();
        fprintf(stderr, "%s in shape (%d %d %d)\n", info.c_str(), dim0, dim1, dim2);
        display_matrix(this->host_w, dim0, dim1, dim2);
    }

    void Blob::show_grad(const string& info){
        this->copy_grad_to_host();
        fprintf(stderr, "%s in shape (%d %d %d)\n", info.c_str(), dim0, dim1, dim2);
        display_matrix(this->host_g, dim0, dim1, dim2);
    }// end Blob

    void beam_entry::init(){
        for(int i=0; i<_beam_size;i++){ parent_indices[i]=i;}
        for(int i=0; i<_beam_size ;i++){ word_indices[i]=i;}
    }

    void inverse_trace(beam_entry* x, int length) {
        for (int i = length - 1; i > 0; i--) {
            beam_entry xi = x[i];
            for (int j = 0; j < xi._beam_size; j++) {
                int parent_idx = xi.parent_indices[j];
                x[i - 1].word_indices[j] = x[i - 1].word_indices[parent_idx];
            }
        }
    }
} // end seq2seq
