CUDA_VISIBLE_DEVICES=4,5,6,7 \
python examples/siamese_loss.py -b 256 -d market1501 -a resnet50 --lr 0.01 --step-size 40 --epochs 100 --features 2048 \
	--logs-dir logs/siamese-loss/market1501-resnet50/lr0.01