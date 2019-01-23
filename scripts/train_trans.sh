CUDA_VISIBLE_DEVICES=4,5,6,7 \
python examples/softmax_loss_transfer.py -ds market1501 -dt dukemtmc \
	--num-instances 0 --lr 0.009 --epochs 50 --ss 20 -b 64 --features 256 --combine-trainval \
	--logs-dir logs/softmax-loss-transfer/market2duke/fea256-bs64-ins0
