# Set Devices   |  distrubuted script | Num of GPUS |  Path of data | 
CUDA_VISIBLE_DEVICES=2,3 ./distributed_train.sh 2 /data/ImageNet --model DNA_a  --sched cosine --epochs 150 --warmup-epochs 5 --drop 0.2 --lr 0.4 --batch-size 64 -j 8 --num-classes 1000 --model-ema 
