CUDA_VISIBLE_DEVICES=4,5,6,7 \
python examples/softmax_loss_distributed.py --dataset dukemtmc --height 256 --width 128 -a resnet101 \
	--num-instances 7 --lr 0.01 --epochs 70 --step-size 20 -b 112 -j 16 --features 512 \
	--logs-dir logs/softmax-loss/dukemtmc-resnet101/ins5-f512-ss20-ra0.5 --dist-url 'tcp://10.1.72.39:8846' --world-size 1 --rank 0
