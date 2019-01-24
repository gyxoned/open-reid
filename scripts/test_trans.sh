CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/softmax_loss_transfer.py -ds market1501 -dt dukemtmc -a resnet50 \
	-b 256 --features 256 --combine-trainval --evaluate \
	--resume logs/softmax-loss-transfer/market2duke/fea256-bs64-ins4/model_best.pth.tar
