# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python examples/softmax_loss_transfer.py -ds market1501 -dt dukemtmc -a resnet_ibn50a \
# 	-b 256 --features 256 --combine-trainval --evaluate \
# 	--resume logs/softmax-loss-transfer/market2duke/fea256-bs64-ins4-resnet_ibn50a/model_best.pth.tar

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/softmax_loss_joint.py -ds market1501 -dt dukemtmc -a resnet_ibn50b \
	-b 256 -j 4 --features 256 --combine-trainval --evaluate \
	--resume logs/softmax-loss-joint/market+duke/fea256-bs64-ins4-resnet_ibn50b/model_best.pth.tar