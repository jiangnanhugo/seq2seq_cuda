device_id=1
data_set=data/chat
save_model_dir=log/"${data_set}"_"${device_id}"
max_source_len=100
max_target_len=100
mkdir -p "${save_model_dir}"
echo "${save_model_dir}"

nohup ./seq2seq -device ${device_id}  \
                -train_data_dir ${data_set}  \
                -save_model_dir "${save_model_dir}"  \
                -emb_size 300  -hidden_size 300 -max_iter 20000000  \
                -save_per_iter 68000  -batch_size 32  -checkpoint_per_iter 1000 \
                -lr 0.01  -lr_decay 0.998  -max_source_len ${max_source_len} \
                -max_target_len ${max_target_len} > "${save_model_dir}.log" &
