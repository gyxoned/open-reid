CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/softmax_loss.py --dataset dukemtmc -a resnet_ibn50a \
	--num-instances 4 --lr 0.01 --epochs 50 --step-size 20 -b 64 --features 256 \
	--logs-dir logs/softmax-loss/pt1.1
