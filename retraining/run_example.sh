#DNA_a/ DNA_b / DNA_c / DNA_d
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 /data/ImageNet  --model DNA_*  \
--epochs 500  --warmup-epochs 5  --batch-size 128 --lr 0.064 --opt rmsproptf --opt-eps 0.001 --sched step --decay-epochs 3 --decay-rate 0.963 --color-jitter 0.06 --drop 0.2  -j 8 --num-classes 1000 --model-ema
