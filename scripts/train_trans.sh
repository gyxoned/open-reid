CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/softmax_loss_transfer.py -ds market1501 -dt dukemtmc \
	--num-instances 4 --lr 0.01 --epochs 50 --ss 20 -b 64 -j 8 --features 256 \
	--logs-dir logs/softmax-loss-transfer/market2duke/fea256-bs64-ins4-abn
