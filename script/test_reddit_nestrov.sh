device_id=2
data_set=data/reddit
max_source_len=100
max_target_len=100
beam_size=10
emb_size=100
hidden_size=100
is_train=false
load_model_dir=checkpoint/"${data_set}"_"${device_id}"/epoch_6

./seq2seq  --is_train=false \
    -device ${device_id}  \
    -train_data_dir ${data_set} \
    -test_data_dir ${data_set}  \
    -load_model_dir "${load_model_dir}"  \
    -emb_size ${emb_size}  -hidden_size ${hidden_size}  \
    -batch_size 32 -max_source_len 50 \
    -max_target_len 50
