#include "data_reader.h"
#include "init.h"
// #include "rnn.h"
#include "model.h"
#include <gflags/gflags.h>
#include <sys/stat.h>
#include <ctime>

DEFINE_double(lr, 0.5, "learning rate");
DEFINE_double(lr_decay, 0.999, "lr decay");
DEFINE_int32(max_epoch, 20, "max epoch to train");
DEFINE_int32(batch_size, 32, "Mini-batch size");
DEFINE_int32(beam_size, 10, "beam search size");
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

        cerr<< "source dict size: " << reader.source_dict_size() <<  \
            ", target dict size: " << reader.target_dict_size() << endl;

        int max_encoder_len = FLAGS_max_source_len;
        // target will addes eos and go_id
        int max_decoder_len = FLAGS_max_target_len + 2;

        reader.load_data(FLAGS_train_data_dir + "/train.txt", FLAGS_batch_size, FLAGS_max_source_len, FLAGS_max_target_len);

        //================initialize model================
        Seq2SeqModel model;
        model.set_param(reader.source_dict_size(), reader.target_dict_size(), FLAGS_batch_size, FLAGS_emb_size, FLAGS_hidden_size);

        model.init_train(max_encoder_len, max_decoder_len, FLAGS_loss_type, FLAGS_opt_type, FLAGS_lr);
        fprintf(stderr, "init ended\n");

        Blob encoder_input, decoder_input, decoder_target;

        encoder_input.set_dim(max_encoder_len,  FLAGS_batch_size);
        decoder_input.set_dim(max_decoder_len, FLAGS_batch_size);
        decoder_target.set_dim(max_decoder_len, FLAGS_batch_size);

        encoder_input.malloced();
        decoder_input.malloced();
        decoder_target.malloced();

        //start training
        fprintf(stderr, "start training....\n");

        for (int epoch = 0; epoch < FLAGS_max_epoch; ++epoch){
            int iter=0;
            float sum_loss = 0.0;
            bool ret = false;
            clock_t start = clock();
            // std::cerr << "epoch: " << epoch+1 << '\n';
            while((ret = reader.get_batch(&encoder_input, &decoder_input, &decoder_target))!=false){
                iter+=1;
                // std::cerr << "after get_batch" << '\n';
                model.inc_timestep();
                sum_loss += model.forward(&encoder_input, &decoder_input, &decoder_target);
                // std::cerr << "after model.forward" << '\n';
                model.backward(&encoder_input, &decoder_input, &decoder_target);
                // std::cerr << "after backeard" << '\n';
                model.clip_gradients(FLAGS_max_gradient_norm);
                model.optimize(&encoder_input, &decoder_input);
                // std::cout << "finished one batch training" << '\n';
            }
            // print loss over epoch
            float loss = sum_loss / iter;
            fprintf(stdout, "epoch=%d, iter=%d, lr=%6f, loss=%6f, ppl=%6f\t", epoch + 1, iter, model.optimizer._lr, loss , exp(loss));
            model.set_lr_decay(FLAGS_lr_decay);
            reader.shulffle_and_bucket();                                       // reload dataset
            model.save_model(FLAGS_save_model_dir + "/epoch_" + to_string(epoch + 1)); // save current model.
            double used=(clock() - start) / (double)CLOCKS_PER_SEC;
            fprintf(stdout, "time=%d sec\n", int(used));
        }
    }

    // TODO: using sampled softmax to reduce gpu memory cosuming on softmax compute
    void run_inference(){
        //===========prepare data============
        DataReader reader;
        reader.load_vocab(FLAGS_train_data_dir + "/source.vocab", FLAGS_train_data_dir + "/target.vocab");

        int max_encoder_len = FLAGS_max_source_len;
        // target will addes eos and go_id
        int max_decoder_len = FLAGS_max_target_len + 2;
        int batch_size = 1;
        int beam_size = FLAGS_beam_size;

        fprintf(stderr, "source dict size : %d, target dict size: %d\n", reader.source_dict_size(), reader.target_dict_size());
        reader.load_data(FLAGS_train_data_dir + "/test.txt", batch_size, FLAGS_max_source_len, FLAGS_max_target_len);

        //================initialize model================
        Seq2SeqModel model;

        model.set_param(reader.source_dict_size(), reader.target_dict_size(), batch_size, FLAGS_emb_size, FLAGS_hidden_size);
        model.init_inference(max_encoder_len, beam_size);
        fprintf(stderr, "init ended\n");

        if (FLAGS_load_model_dir.size() > 0){
            cerr << "loading models from checkpoints: " << FLAGS_load_model_dir << "\n";
            model.load_model(FLAGS_load_model_dir);
        }

        Blob encoder_input, decoder_input;

        encoder_input.set_dim(max_encoder_len, batch_size);
        decoder_input.set_dim(1, batch_size);

        encoder_input.malloced();
        decoder_input.malloced();

        bool ret = false;
        while((ret = reader.get_batch(&encoder_input, &decoder_input)) != false){
            fprintf(stderr, "calc loss....\n");
            model.encode(&encoder_input);
            for(int t = 0 ; t < max_decoder_len; ++t){
                model.step(&decoder_input, t==0? true : false);
            }
        }
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

    google::ShutDownCommandLineFlags();
    return 0;
}
