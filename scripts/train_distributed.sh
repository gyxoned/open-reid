CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/softmax_loss_distributed.py --dataset dukemtmc --height 256 --width 128 -a resnet50 \
	--num-instances 4 --lr 0.01 --epochs 70 --step-size 40 -b 64 -j 16 --features 512 \
	--logs-dir logs/softmax-loss/dukemtmc-resnet50/ins4-f512-ss40 --dist-url 'tcp://10.1.72.207:8846' --world-size 1 --rank 0
