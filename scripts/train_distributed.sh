CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/softmax_loss_distributed.py --dataset market1501 --height 256 --width 128 -a resnet101 \
	--num-instances 4 --lr 0.009 --epochs 70 --step-size 20 -b 64 -j 16 --features 512 \
	--logs-dir logs/softmax-loss/market1501-resnet101/ins4-f512 --dist-url 'tcp://10.1.72.207:8845' --world-size 1 --rank 0
