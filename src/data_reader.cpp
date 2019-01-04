#include "data_reader.h"
#include <iostream>
#include <algorithm>
#include <fstream>

namespace seq2seq {
    const int DataReader::PAD_ID = 0;
    const int DataReader::GO_ID = 1;
    const int DataReader::EOS_ID = 2;
    const int DataReader::UNK_ID = 3;
    const string DataReader::_pattern=" <EOS>#TAB#";

    seq_pair::seq_pair(const vector<int>& source_vec, const vector<int>& target_vec) {
        source_idx = new int[source_vec.size()];
        assert(source_idx != NULL);
        target_idx = new int[target_vec.size()];
        assert(target_idx != NULL);

        for (unsigned int i = 0; i < source_vec.size(); ++i) {
            source_idx[i] = source_vec[i];
        }

        for (unsigned int i = 0; i < target_vec.size(); ++i) {
            target_idx[i] = target_vec[i];
        }
    }

    void DataReader::load_data(const string& source_file, const int batch_size, const unsigned int max_source_len, const unsigned int max_target_len){
        _batch_size = batch_size;
        _max_source_len = max_source_len;
        _max_target_len = max_target_len;

        ifstream source_f(source_file);
        assert(source_f.good());

        string line;

        vector<int> source_idx, target_idx;
        while (getline(source_f, line)){
            int pos = line.find(_pattern);
            string source = line.substr(0, pos);
            string target = line.substr(pos + _pattern.length());

            this->str_to_idx(source, source_idx, _source_dict);
            this->str_to_idx(target, target_idx, _target_dict);

            if (source_idx.size() > _max_source_len || target_idx.size() > _max_target_len){
                continue;
            }

            target_idx.push_back(EOS_ID);                                       // insert a eos into target_ids
            _all_data.push_back(shared_ptr<seq_pair>(new seq_pair(source_idx, target_idx)));
        }

        size_t data_size = _all_data.size();
        fprintf(stderr, "pairs loaded: %ld\n", data_size);

        _data_idx.reset(new size_t[data_size]);

        for (unsigned int i = 0; i < data_size; ++i) {                          // build index
            _data_idx.get()[i] = i;
        }
        this->shulffle_and_bucket();
    }

    bool DataReader::prefetch() {
        _prefetched_examples.clear();
        _prefetch_cursor = 0;

        if (_cursor >= _all_data.size()) {
            return false;
        }
        size_t total_fetch = _batch_size * prefetch_batch_count;
        // pick from all_data sequentially
        for (size_t k=0; k < total_fetch; ++k){
            if(_cursor <_all_data.size()){
                _prefetched_examples.push_back(_data_idx.get()[_cursor]);
                ++_cursor;
            }else{
                size_t rand_idx = rand() % _all_data.size();                    // pick randomly if no enough data left
                _prefetched_examples.push_back(rand_idx);
            }
        }
        // sort the prefetched examples
        sort(_prefetched_examples.begin(), _prefetched_examples.end(),
        [this](size_t a, size_t b) {
            if (this->_all_data[a]->source_len < this->_all_data[b]->source_len) {
                return true;
            }else if (this->_all_data[a]->source_len == this->_all_data[b]->source_len) {
                return this->_all_data[a]->target_len < this->_all_data[b]->target_len;
            }else{
                return false;
            }
        });
        return true;
    }

    void DataReader::shulffle_and_bucket() {
        _cursor = 0;
        cerr << "shuffling data....." << endl;
        random_shuffle(_data_idx.get(), _data_idx.get() + _all_data.size());        // for shuffling, we just shuffle the index.
        prefetch();
        cerr << "bucket size:" << _prefetched_examples.size() << endl;
    }

    // encoder_input : encoder_size * batch,
    // decoder_input, decoder_target : decoder_size * batch
    bool DataReader::get_batch(Blob* encoder_input, Blob* decoder_input, Blob* decoder_target) {
        cerr << "cursor:" << _prefetch_cursor << " example_size: " << _prefetched_examples.size() << endl;

        size_t batch_end = _prefetch_cursor + _batch_size;
        int source_length = _all_data[_prefetched_examples[batch_end - 1]]->source_len;

        int target_length = -1;
        for(unsigned int k = 0 ; k < _batch_size; k++){
            int length = this -> _all_data[_prefetch_cursor + k]->target_len + 1;
            target_length = (length > target_length ? length : target_length);
        }

        float* en_input = encoder_input->host_data;
        float* de_input = decoder_input->host_data;
        float* de_target = decoder_target->host_data;

        memset(en_input, static_cast<float>(PAD_ID), source_length * _batch_size * sizeof(float));
        memset(de_input, static_cast<float>(PAD_ID), target_length * _batch_size * sizeof(float));
        memset(de_target, static_cast<float>(PAD_ID), target_length * _batch_size * sizeof(float));

        for (unsigned int i = 0; i < _batch_size; ++i) {
            int sent_idx = _prefetched_examples[_prefetch_cursor + i];
            const seq_pair* pair_t = _all_data[sent_idx].get();

            // reverse encoder input
            int k= (source_length - pair_t->source_len) * _batch_size;
            for (int j = pair_t->source_len - 1; j >= 0; --j) {
                en_input[k] = static_cast<float>(pair_t->source_idx[j]);
                k += _batch_size;
            }

            // insert GO_ID into decoder begining
            k = i;
            de_input[k] = static_cast<float>(GO_ID);
            k += _batch_size;
            // copy decoder real inputs
            for (unsigned int j = 0; j < pair_t->target_len ; ++j) {
                de_input[k] = static_cast<float>(pair_t->target_idx[j]);
                de_target[k - _batch_size] = de_input[k];       // decoder_target is a shift of decoder_input
                k += _batch_size;
            }
        }

        encoder_input->set_dim(source_length, _batch_size);
        decoder_input->set_dim(target_length, _batch_size);
        decoder_target->set_dim(target_length, _batch_size);

        fprintf(stderr, "source length: %d, target_length:%d\n", source_length, target_length);

        encoder_input->copy_data_to_device();
        decoder_input->copy_data_to_device();
        decoder_target->copy_data_to_device();

        _prefetch_cursor += _batch_size;
        if (_prefetch_cursor >= _prefetched_examples.size()) {
            prefetch();
        }
        if(_cursor >= _all_data.size()){
            return false;
        }
        return true;
    }

    // encoder_input : encoder_size * batch,
    // decoder_input, decoder_target : decoder_size * batch
    bool DataReader::get_batch(Blob* encoder_input, Blob* decoder_input){
        cerr << "cursor:" << _prefetch_cursor << " example_size: " << _prefetched_examples.size() << endl;

        size_t batch_end = _prefetch_cursor + _batch_size;
        int source_length = _all_data[_prefetched_examples[batch_end - 1]]->source_len;

        int target_length = -1;
        for(unsigned int k = 0 ; k < _batch_size; k++){
            int length = this -> _all_data[_prefetch_cursor + k]->target_len + 1;
            target_length = (length > target_length ? length : target_length);
        }

        float* en_input = encoder_input->host_data;
        float* de_input = decoder_input->host_data;

        memset(en_input, static_cast<float>(PAD_ID), source_length * _batch_size * sizeof(float));
        memset(de_input, static_cast<float>(PAD_ID), _batch_size * sizeof(float));

        for (unsigned int i = 0; i < _batch_size; ++i) {
            int sent_idx = _prefetched_examples[_prefetch_cursor + i];
            const seq_pair* pair_t = _all_data[sent_idx].get();

            // reverse encoder input
            int k = (source_length - pair_t->source_len) * _batch_size;
            for (int j = pair_t->source_len - 1; j >= 0; --j) {
                en_input[k] = static_cast<float>(pair_t->source_idx[j]);
                k += _batch_size;
            }

            // insert GO_ID into decoder begining
            k = i;
            de_input[k] = static_cast<float>(GO_ID);
        }

        encoder_input->set_dim(source_length, _batch_size);
        decoder_input->set_dim(target_length, _batch_size);

        fprintf(stderr, "source length: %d, target_length:%d\n", source_length, target_length);

        encoder_input->copy_data_to_device();
        decoder_input->copy_data_to_device();

        _prefetch_cursor += _batch_size;
        if (_prefetch_cursor >= _prefetched_examples.size()) {
            prefetch();
        }
        if(_cursor >= _all_data.size()){
            return false;
        }
        return true;
    }

    void DataReader::display_all_data() {
        for (const auto& item : _all_data) {
            fprintf(stderr, "source: ");
            display_matrix(item->source_idx, 1, item->source_len);
            fprintf(stderr, "target: ");
            display_matrix(item->target_idx, 1, item->target_len);
        }
    }

    void DataReader::load_vocab(const string& source_vocab, const string& target_vocab) {
        _source_vocab = source_vocab;
        _target_vocab = target_vocab;
        this->load_dict(source_vocab, _source_dict, _rev_source_dict_vec);
        this->load_dict(target_vocab, _target_dict, _rev_target_dict_vec);
    }

    void DataReader::load_dict(const string& vocab_file,unordered_map<string, int>& dict, vector<string>& rev_dict) {
        ifstream file(vocab_file);
        if (!file.good()) {
            cerr<<"error dict file:"<<vocab_file<<endl;
            exit(-1);
        }

        string line;
        int index = 0;
        while (getline(file, line)) {
            if (dict.count(line) > 0) {
                fprintf(stderr, "duplicated entry:[%s] in file %s\n", line.c_str(), vocab_file.c_str());
                exit(-1);
            }
            dict[line] = index++;
            rev_dict.push_back(line);
        }

        // check top 4 entries
        if (rev_dict.size() < 4 || rev_dict[0] != "_PAD"|| rev_dict[1] != "_GO"|| rev_dict[2] != "_EOS"|| rev_dict[3] != "_UNK") {
            fprintf(stderr, "top four words in dict should be: _PAD, _GO, _EOS, _UNK\n");
            exit(-1);
        }
    }

    void DataReader::str_to_idx(const string& str,vector<int>& result, const unordered_map<string,int>& dict) {
        vector<string> tokens;
        split(str, tokens);

        result.resize(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (dict.count(tokens[i]) > 0) {
                result[i] = dict.at(tokens[i]);
            }else {
                fprintf(stderr, "[%s] mapped to UNK in [%s]\n", tokens[i].c_str(), str.c_str());
                result[i] = UNK_ID;
            }
        }
    }

} // namespace seq2seq
