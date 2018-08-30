CUDA_VISIBLE_DEVICES=4,5,6,7 \
python examples/softmax_loss.py --dataset dukemtmc \
	--num-instances 7 --lr 0.01 --epochs 70 --ss 20 -b 112 --features 256 \
	--logs-dir logs/softmax-loss/dukemtmc-resnet50/lr0.01-baseline
