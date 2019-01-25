# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python examples/softmax_loss_transfer.py -ds market1501 -dt dukemtmc -a resnet_ibn50b \
# 	--num-instances 7 --lr 0.009 --epochs 50 --ss 20 -b 112 --features 256 --combine-trainval \
# 	--logs-dir logs/softmax-loss-transfer/market2duke/fea256-bs112-ins7-resnet_ibn50b

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python examples/softmax_loss_joint.py -ds market1501 -dt dukemtmc -a resnet_ibn50b \
	--num-instances 4 --lr 0.009 --epochs 50 --ss 20 -b 64 --features 256 --combine-trainval \
	--logs-dir logs/softmax-loss-joint/market+duke/fea256-bs64-ins4-resnet_ibn50b