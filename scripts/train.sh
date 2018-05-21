srun -p OCR --gres=gpu:4 -n1 --ntasks-per-node=4 \
	python examples/rw_loss.py --combine-trainval -d dukemtmc \
				--logs-dir logs/finetune/dukemtmc/ --lr 0.0001 \
				--epochs 50 --ss 20 --weight-decay 0 \
				--retrain /mnt/lustre/geyixiao/ECCV2018/model_zoo/siamese-loss/dukemtmc/model_best.pth.tar
