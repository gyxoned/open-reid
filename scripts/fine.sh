CUDA_VISIBLE_DEVICES=4,5,6,7 \
python examples/finetune.py -d msmt17 \
		--logs-dir logs/siamese-loss/msmt17-resnet50-finetune/ba0.0001-em0.0001 --lr 0.0001 \
		--epochs 50 --ss 20 --weight-decay 0 \
		--retrain logs/siamese-loss/msmt17-resnet50/ba0.01-emb0.01-ss40/model_best.pth.tar