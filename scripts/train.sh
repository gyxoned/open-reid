CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/oim_loss.py --dataset cuhk03 --oim-scalar 30 \
	--num-instances 7 --lr 0.009 --epochs 70 -b 112 --features 256 \
	--logs-dir logs/oim-loss/cuhk03-resnet50/lr0.009
