CUDA_VISIBLE_DEVICES=4,5,6,7 \
python examples/oim_loss.py --dataset market1501 --oim-scalar 30 \
	--num-instances 7 --lr 0.0009 --epochs 70 -b 112 --features 256 \
	--logs-dir logs/oim-loss/market1501-resnet50/lr0.0009val
