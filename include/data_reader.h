#ifndef SEQ2SEQ_INCLUDE_DATA_READER_H
#define SEQ2SEQ_INCLUDE_DATA_READER_H

#include "common.h"
#include "init.h"
#include <string>
#include <map>
#include <unordered_map>

namespace seq2seq {
    // brief: a training example
    struct TrainPair {
        TrainPair(): source_ids(NULL), target_ids(NULL), source_len(0), target_len(0) {}
        TrainPair(const vector<int>& source_vec, const vector<int>& target_vec);
        ~TrainPair() {
            if (source_ids != NULL) {delete []source_ids;}
            if (target_ids != NULL) {delete []target_ids;}
        }

        int* source_ids, *target_ids;
        unsigned int source_len, target_len;
    };

    // this can be protected by a shared_ptr
    class DataReader {
        public:
            static const int PAD_ID, GO_ID, EOS_ID, UNK_ID;      // 3
            static const string _pattern;
        public:
            explicit DataReader() {}
            ~DataReader() {}

            void load_vocab(const string& source_vocab, const string& target_vocab);

            inline int source_dict_size() {return (int)_rev_source_dict_vec.size();}

            inline int target_dict_size() {return (int)_rev_target_dict_vec.size();}

            void load_data(const string& source_file,
                    const int batch_size = 32, const unsigned int max_source_len = 50, const unsigned int max_target_len = 50);

            /*
             * logic inside this routine:
             *    1) pick a bucket according to buckets_scale
             *    2) randomly pick batch_size examples from this bucket
             *    3) pad training pairs to
             *
             * @param [in] batch_size
             * @param [out] encoder_input in shape seq_length * batch, A B C
             * @param [out] decoder_input in shape seq_length * batch, GO W X Y Z
             * @param [out] decoder_target in shape seq_length * batch, W X Y Z EOS
             */
            bool get_batch(Blob* encoder_input,  Blob* decoder_input, Blob* decoder_target);
            void shulffle_and_bucket();
            void display_all_data();

        private:
            // data structure: _all_data really stores all the data loaded from training files
            // _buckets is the user specified bucket list
            // _data_set size is same as _buckets, each is an array of index of training examples
            // that should stay in that bucket
            vector<shared_ptr<TrainPair> > _all_data;
            unsigned int _max_source_len, _max_target_len, _batch_size;

            shared_ptr<size_t> _data_idx; // cursor points to _example_idx, all the data index
            size_t _cursor;

            // prefetch more batches, then, sort and return a batch
            vector<size_t> _prefetched_examples;
            // prefetch this much batch examples
            static const size_t prefetch_batch_count = 512;
            //cursor points to prefetched_examples
            size_t _prefetch_cursor;

            bool prefetch();

            void load_dict(const string& vocab_file, unordered_map<string, int>& dict, vector<string>& rev_dict);
            void str_to_ids(const string& str, vector<int>& result, const unordered_map<string, int>& dict);

            unordered_map<string, int> _source_dict_map, _target_dict_map;

            vector<string> _rev_source_dict_vec, _rev_target_dict_vec;

            string _source_vocab, _target_vocab, _source_file, _target_file;                                                // target training filename
        private:
            DISALLOW_COPY_AND_ASSIGN(DataReader);
    };
} // namespace seq2seq

#endif
