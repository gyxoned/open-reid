CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/softmax_loss.py --dataset dukemtmc \
	--num-instances 7 --lr 0.05 --epochs 70 --ss 20 -b 112 --features 256 \
	--logs-dir logs/softmax-loss/dukemtmc-resnet50/lr0.05
