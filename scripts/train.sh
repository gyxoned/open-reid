CUDA_VISIBLE_DEVICES=6,7 \
python examples/softmax_loss.py -d market1501 -a resnet50 --lr 0.1 --step-size 40 --epochs 100 --features 1024 \
	--logs-dir logs/softmax-loss/market1501-resnet50/embd1024-lr0.1