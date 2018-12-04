device_id=1
data_set=../data/chat
save_model_dir=../checkpoint/"${data_set}"_"${device_id}"
max_source_len=100
max_target_len=100
mkdir -p "${save_model_dir}"
echo "${save_model_dir}"

../seq2seq -device 1  \
    -train_data_dir ${data_set}  \
    -save_model_dir "${save_model_dir}"  \
    -emb_size 300  -hidden_size 300  \
    -batch_size 32  -checkpoint_per_iter 100 \
    -lr 0.1  -lr_decay 0.998  -max_source_len 50 \
    -max_target_len 50
