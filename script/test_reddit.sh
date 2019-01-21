device_id=2
data_set=data/reddit
max_source_len=100
max_target_len=100
beam_size=10
is_train=false
load_model_dir=checkpoint/"${data_set}"_"${device_id}"/epoch_6
mkdir -p "${save_model_dir}"
echo "${save_model_dir}"

./seq2seq -device ${device_id}  \
    -test_data_dir ${data_set}  \
    -load_model_dir "${load_model_dir}"  \
    -emb_size 300  -hidden_size 300  \
    -batch_size 32 -max_source_len 50 \
    -max_target_len 50 -is_train ${is_train}
