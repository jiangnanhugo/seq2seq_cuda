#include "data_reader.h"
#include "all_computes.h"
// #include "rnn.h"
#include "model.h"
#include <gflags/gflags.h>


DEFINE_double(lr, 0.5, "learning rate");
DEFINE_double(lr_decay, 0.999, "lr decay");
DEFINE_int32(max_epoch, 50, "max epoch to train");
DEFINE_int32(batch_size, 32, "Mini-batch size");
DEFINE_int32(emb_size, 512, "Embedding size");
DEFINE_int32(hidden_size, 1024, "Hidden size");
DEFINE_bool(is_train, true, "train or inference");
DEFINE_string(train_data_dir, "", "which folder to load training data");
DEFINE_string(save_model_dir, "", "which folder to save trained model");
DEFINE_string(load_model_dir, "", "which folder to load model (leave it empty for cold start)");
DEFINE_int32(checkpoint_per_iter, 200, "How many iters per checkpoint.");
DEFINE_double(max_gradient_norm, 5.0, "Clip gradients to this norm.");
DEFINE_int32(device, 0, "which device to use");
DEFINE_int32(max_source_len, 50, "max source language length");
DEFINE_int32(max_target_len, 50, "max target language length");
DEFINE_string(loss_type,"cross_entropy","focal loss/cross_entropy loss");
DEFINE_string(opt_type,"sgd","sgd/sgd_m loss");
namespace seq2seq{
    void run_train(){
        //===========prepare data============
        DataReader reader;
        reader.load_vocab(FLAGS_train_data_dir + "/source.vocab", FLAGS_train_data_dir + "/target.vocab");

        cerr<< "source dict size: "<<reader.source_dict_size()<< ", target dict size: "<<reader.target_dict_size()<<endl;

        int max_encoder_len = FLAGS_max_source_len;
        // target will addes eos and go_id
        int max_decoder_len = FLAGS_max_target_len + 2;

        reader.load_data(FLAGS_train_data_dir + "/train.txt", FLAGS_batch_size, FLAGS_max_source_len, FLAGS_max_target_len);

        //================initialize model================
        Seq2SeqModel model;
        model.set_param(reader.source_dict_size(), reader.target_dict_size(), FLAGS_batch_size, FLAGS_emb_size, FLAGS_hidden_size);

        model.init(max_encoder_len, max_decoder_len,FLAGS_loss_type, FLAGS_opt_type, FLAGS_lr);
        fprintf(stderr, "init ended\n");

        if (FLAGS_load_model_dir.size() > 0){
            fprintf(stderr, "loading models from checkpoints......\n");
            model.load_model(FLAGS_load_model_dir);
        }

        Blob encoder_input, decoder_input, decoder_target;

        encoder_input.set_dim(max_encoder_len,  FLAGS_batch_size);
        decoder_input.set_dim(max_decoder_len, FLAGS_batch_size);
        decoder_target.set_dim(max_decoder_len, FLAGS_batch_size);

        encoder_input.malloc();
        decoder_input.malloc();
        decoder_target.malloc();

        //start training
        fprintf(stderr, "start training....\n");
        for (int epoch = 0; epoch < FLAGS_max_epoch; ++epoch){
            int iter=0;
            float sum_loss = 0.0;
            bool ret=false;
            while((ret=reader.get_batch(&encoder_input, &decoder_input, &decoder_target))!=false){
                iter++;
                sum_loss += model.forward(&encoder_input, &decoder_input, &decoder_target);
                model.backward(&encoder_input, &decoder_input, &decoder_target);
                model.clip_gradients(FLAGS_max_gradient_norm);
                model.optimize(&encoder_input, &decoder_input);
            }
            // print loss over epoch
            float loss=sum_loss / iter;
            fprintf(stdout, "epoch=%d, iter=%d, lr=%6f, loss=%6f, ppl=%6f\n", epoch + 1, iter, model.lr, loss , exp(loss));
            model._lr *= FLAGS_lr_decay;
            reader.shulffle_and_bucket();                                       // reload dataset
            model.save_model(FLAGS_save_model_dir + "/epoch_" + to_string(epoch + 1)); // save current model.
        }
    }

    // TODO: using sampled softmax to reduce gpu memory cosuming on softmax compute
    void run_inference(){
        //===========prepare data============
        DataReader reader;
        string source_vocab = FLAGS_train_data_dir + "/source.vocab";
        string target_vocab = FLAGS_train_data_dir + "/target.vocab";
        reader.load_vocab(source_vocab,target_vocab);

        fprintf(stderr, "source dict size : %d, target dict size: %d\n",
                reader.source_dict_size(), reader.target_dict_size());

        int max_encoder_len = FLAGS_max_source_len;
        // target will addes eos and go_id
        int max_decoder_len = FLAGS_max_target_len + 2;
        reader.load_all_data(FLAGS_train_data_dir + "/train.txt",
                FLAGS_batch_size, FLAGS_max_source_len, FLAGS_max_target_len);

        //================initialize model================
        Seq2SeqModel model;
        model.set_param(reader.source_dict_size(), reader.target_dict_size(),FLAGS_lr,FLAGS_batch_size, FLAGS_emb_size, FLAGS_hidden_size);

        model.init(max_encoder_len, max_decoder_len,"cross_entropy", FLAGS_opt_type);
        fprintf(stderr, "init ended\n");

        if (FLAGS_load_model_dir.size() > 0){
            model.load_model(FLAGS_load_model_dir.c_str());
        }

        Blob encoder_input, decoder_input, decoder_target;

        encoder_input.set_dim(max_encoder_len, FLAGS_batch_size);
        decoder_input.set_dim(max_decoder_len, FLAGS_batch_size);
        decoder_target.set_dim(max_decoder_len, FLAGS_batch_size);

        encoder_input.malloc();
        decoder_input.malloc();
        decoder_target.malloc();

        vector<float> losses;
        //====================start optimization===========
        int checkpoint = 0;
        float checkpoint_avg_loss = 0.0;

        bool ret=false;
        int iter=0;
        while((ret=reader.get_batch(&encoder_input, &decoder_input, &decoder_target))!=false){
            fprintf(stderr, "calc loss....\n");
            float iter_loss = model.forward(&encoder_input, &decoder_input, &decoder_target);
            checkpoint_avg_loss += iter_loss;

            if ((iter + 1) % FLAGS_checkpoint_per_iter == 0){
                ++checkpoint;
                float loss = checkpoint_avg_loss / FLAGS_checkpoint_per_iter;
                checkpoint_avg_loss = 0.0;

                float perplexity = exp(loss);
                fprintf(stderr, " iter= %d, loss = %6f, ppl = %6f\n",
                        iter, loss, perplexity);

                losses.push_back(loss);
            }
        }
        string to_save_dir = FLAGS_save_model_dir + "/epoch_" ;
        mkdir(to_save_dir.c_str(), 0777);
        fprintf(stderr, "saving model to %s\n", to_save_dir.c_str());
        model.save_model(to_save_dir.c_str());
    }
}// namespace seq2seq

int main(int argc, char **argv){
    //  srand(time(NULL));
    if (argc < 2){
        fprintf(stderr, "use --help for details\n");
        exit(-1);
    }
    google::SetUsageMessage("seq2seq training");
    google::SetVersionString("0.0.1");
    google::ParseCommandLineFlags(&argc, &argv, true);
    cudaErrCheck(cudaSetDevice(FLAGS_device));
    seq2seq::g_assert=GlobalAssets::instance();

    if (FLAGS_train_data_dir.size() == 0 || FLAGS_save_model_dir.size() == 0){
        fprintf(stderr, "you must set train_data_dir and save_model_dir\n"
                "use --help for details\n");
        exit(-1);
    }

    struct stat sb;
    if (stat(FLAGS_save_model_dir.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)){
        fprintf(stderr, "%s is not a valid path\n", FLAGS_save_model_dir.c_str());
        exit(-1);
    }
    if( FLAGS_is_train){
        seq2seq::run_train();
    }else{
        seq2seq::run_inference();
    }
    // seq2seq::g_assert.free();
    google::ShutDownCommandLineFlags();
    return 0;
}
