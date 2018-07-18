CUDA_VISIBLE_DEVICES=2,3 \
python examples/siamese_loss.py -b 256 -d msmt17 -a resnet50 --features 2048 \
	--evaluate --resume logs/siamese-loss/msmt17-resnet50/lr0.01/model_best.pth.tar