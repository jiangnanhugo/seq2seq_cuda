#include "blob.h"

namespace seq2seq{
    void Blob::malloced() {
        int shape = size();
        host_data = (float*)malloc(shape * sizeof(float));
        assert(host_data != NULL);
        cudaErrCheck(cudaMalloc((void**)&device_data, shape * sizeof(float)));
        cudaErrCheck(cudaMemset(device_data, 0.0, shape * sizeof(float)));

        host_diff = (float*)malloc(shape * sizeof(float));
        assert(host_diff != NULL);
        cudaErrCheck(cudaMalloc((void**)&device_diff, shape * sizeof(float)));
        cudaErrCheck(cudaMemset(device_diff, 0.0, shape * sizeof(float)));

        host_moment1 = (float*)malloc(shape * sizeof(float));
        assert(host_moment1 != NULL);
        cudaErrCheck(cudaMalloc((void**)&host_moment1, shape * sizeof(float)));
        cudaErrCheck(cudaMemset(host_moment1, 0.0, shape * sizeof(float)));

        host_moment2 = (float*)malloc(shape * sizeof(float));
        assert(host_moment2 != NULL);
        cudaErrCheck(cudaMalloc((void**)&host_moment2, shape * sizeof(float)));
        cudaErrCheck(cudaMemset(host_moment2, 0.0, shape * sizeof(float)));
    }

    void Blob::copy_data_to_device() {
        cudaErrCheck(cudaMemcpy(device_data, host_data, size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    void Blob::copy_data_to_host() {
        cudaErrCheck(cudaMemcpy(host_data, device_data, size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    void Blob::copy_diff_to_device() {
        cudaErrCheck(cudaMemcpy(device_diff, host_diff, size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    void Blob::copy_diff_to_host() {
        cudaErrCheck(cudaMemcpy(host_diff, device_diff, size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // saving matrix (ignore dim3) into a text file
    void Blob::savetxt(const string& filename) {
        this->copy_data_to_host();
        const float* data = this->host_data;
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
        fprintf(stderr, "loading %s\n", filename.c_str());
        float* data = this->host_data;
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
        this->copy_data_to_device();
    }

    // for debug purpose
    void Blob::display_data(const string& info){
        this->copy_data_to_host();
        fprintf(stderr, "%s in shape (%d %d %d)\n", info.c_str(), dim0, dim1, dim2);
        display_matrix(this->host_data, dim0, dim1, dim2);
    }

    void Blob::display_diff(const string& info){
        this->copy_diff_to_host();
        fprintf(stderr, "%s in shape (%d %d %d)\n", info.c_str(), dim0, dim1, dim2);
        display_matrix(this->host_diff, dim0, dim1, dim2);
    }
    // end Blob
} // end seq2seq
