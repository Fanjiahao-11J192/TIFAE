set -e
run_idx=$1
pretrained_idx=$2
gpu=$3
transformer_heads=$4
transformer_layers=$5

for i in `seq 1 1 10`;
do

cmd="python train_miss.py --dataset_mode=multimodal_miss --model=TIFAE
--log_dir=./logs --checkpoints_dir=./checkpoints --gpu_ids=$gpu
--A_type=comparE --input_dim_a=130 --norm_method=trn --embd_size_a=128 --conv_dim_a=40 --embd_method_a=maxpool
--V_type=denseface --input_dim_v=342 --embd_size_v=128 --conv_dim_v=40 --embd_method_v=maxpool
--L_type=bert_large --input_dim_l=1024 --embd_size_l=128 --conv_dim_l=40
--transformer_heads=$transformer_heads --transformer_layers=$transformer_layers
--AE_layers=256,128,64 --n_blocks=5 --num_thread=0 --corpus=IEMOCAP
--pretrained_path='checkpoints/CAP_pretrained_TIFAE_AVL_run$pretrained_idx'
--ce_weight=1.0 --mse_weight=2.0 --cycle_weight=2.0 --trans_weight=100
--output_dim=4 --cls_layers=128,128 --dropout_rate=0.5
--niter=60 --niter_decay=60 --verbose --print_freq=10 --in_mem
--batch_size=32 --lr=2e-4 --run_idx=$run_idx --weight_decay=1e-5
--name=TIFAE_IEMOCAP --suffix=block_{n_blocks}_run{run_idx} --has_test
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done