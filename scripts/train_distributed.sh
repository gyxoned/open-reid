CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/softmax_loss_distributed.py --dataset dukemtmc -a resnet_ibn50a \
	--num-instances 4 --lr 0.01 --epochs 70 --step-size 20 -b 64 -j 8 --features 256 \
	--logs-dir logs/softmax-loss/debug --dist-url 'tcp://10.1.72.207:8845' --world-size 1 --rank 0
