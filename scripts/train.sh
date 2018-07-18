CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/siamese_loss.py -b 256 -d msmt17 -a resnet50 --lr 0.01 --step-size 60 --epochs 150 --features 2048 \
	--logs-dir logs/siamese-loss/msmt17-resnet50/ba0.01-emb0.01-ss60