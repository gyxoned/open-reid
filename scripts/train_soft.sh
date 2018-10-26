CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/softmax_loss.py --dataset cuhk03np \
	--num-instances 0 --lr 0.01 --epochs 70 --ss 20 -b 64 --features 256 \
	--logs-dir logs/softmax-loss/cuhk03np-resnet50/nonsampler-bs64
