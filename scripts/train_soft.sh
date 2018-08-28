CUDA_VISIBLE_DEVICES=0,3,6,7 \
python examples/softmax_loss.py --dataset market1501 \
	--num-instances 7 --lr 0.01 --epochs 70 --ss 20 -b 112 --features 256 \
	--logs-dir logs/softmax-loss/market1501-resnet50
