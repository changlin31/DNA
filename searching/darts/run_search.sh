
CUDA_VISIBLE_DEVICES=0  \
./dist_darts_search.sh 8 ${IMAGENET_PATH}$ \
 --model darts --epochs 90  \
 --warmup-epochs 5  --batch-size 128 \
 --lr 0.064 --opt rmsproptf --opt-eps 0.001 \
 --sched step --decay-epochs 3 --decay-rate 0.963 \
 --color-jitter 0.06 --drop 0.2  -j 8 --num-classes 1000  
