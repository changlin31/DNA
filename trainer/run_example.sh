# Set Devices   |  distrubuted script | Num of GPUS |  Path of data | 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 ImageNet --model dna_a  --sched cosine --epochs 150 --warmup-epochs 5 --drop 0.2 --lr 0.4 --batch-size 256 -j 8 --num-classes 1000 --model-ema 
