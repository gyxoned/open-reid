# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python examples/softmax_loss_transfer_dom.py -dt dukemtmc -ds market1501 -a resnet_ibn50a \
# 	--num-instances 4 --lr 0.009 --epochs 50 --ss 20 -b 32 --features 256 --combine-trainval \
# 	--logs-dir logs/softmax-loss-transfer/market2duke/fea256-bs32-ins4-dom

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python examples/softmax_loss_joint.py -ds market1501 -dt dukemtmc -a resnet_ibn50a \
# 	--num-instances 4 --lr 0.009 --epochs 50 --ss 20 -b 64 --features 256 --combine-trainval \
# 	--logs-dir logs/softmax-loss-joint/market+duke/fea256-bs64-ins4-resnet_ibn50a

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/softmax_loss_transfer.py -dt dukemtmc -ds market1501 -a resnet_ibn50a \
	--num-instances 4 --lr 0.0009 --epochs 50 --ss 20 -b 64 --features 256 --combine-trainval \
	--init logs/softmax-loss-transfer/market2duke/fea256-bs64-ins4-dom-resnet_ibn50a/model_best.pth.tar \
	--logs-dir logs/softmax-loss-transfer-cluster/market2duke/fea256-bs64-ins4-resnet_ibn50a-clu100
