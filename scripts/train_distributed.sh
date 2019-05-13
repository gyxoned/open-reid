CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/softmax_loss_distributed.py --dataset aic19 --height 256 --width 256 -a resnet_ibn50a \
	--num-instances 0 --lr 0.01 --epochs 70 --step-size 20 -b 64 -j 16 --features 256 \
	--logs-dir logs/softmax-loss/aic19/ibn50a-ins0-f256 --dist-url 'tcp://10.1.72.207:8845' --world-size 1 --rank 0
