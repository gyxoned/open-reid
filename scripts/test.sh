GPU=$1
ARCH=$2
DATASET=$3
MODEL=$4

CUDA_VISIBLE_DEVICES=${GPU} \
python examples/softmax_loss.py --dataset ${DATASET} -a ${ARCH} \
        -b 256 -j 4 --features 512 --evaluate \
    	--resume ${MODEL} 
