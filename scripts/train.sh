srun -p DSK --gres=gpu:4 -n1 --ntasks-per-node=4 \
python examples/fine.py --combine-trainval -d market1501 \
				--logs-dir logs/finetune/stage2_cat_normal_p128_z256_drop0.2_cn0_L100_M0.0_C10_mar0.0_earser --lr 0.00001 \
				--epochs 50 --ss 20 -b 256 \
				--retrain /mnt/lustre/geyixiao/ECCV2018/pytorch-CycleGAN-and-pix2pix/checkpoints/nips/siamese-loss/market1501/stage2_cat_normal_p128_z256_drop0.2_cn0_L100_M0.0_C10_mar0.0_earser/50_net_reid_encoder.pth
