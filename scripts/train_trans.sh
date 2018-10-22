CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/softmax_loss_transfer.py --dataset market1501 --dataset-ul dukemtmc \
	--num-instances 4 --lr 0.01 --epochs 70 --ss 20 -b 64 --features 256 --combine-trainval \
	--logs-dir logs/softmax-loss-transfer/market2duke
