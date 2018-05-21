srun -p Test --gres=gpu:1 -n1 --ntasks-per-node=1 \
	python examples/pairwise_loss.py --combine-trainval --evaluate -d dukemtmc \
  --resume /mnt/lustre/geyixiao/ECCV2018/pytorch-CycleGAN-and-pix2pix/checkpoints/nips/siamese-loss/dukemtmc/stage2_cat_normal_p128_z256_drop0.2_cn0_L100_M0.0_C10_mar80.0_gauss_share1/5_net_reid_encoder.pth
# 	--resume /mnt/lustre/geyixiao/ECCV2018/pytorch-CycleGAN-and-pix2pix/checkpoints/siamese-loss/market1501/stage2_cat_normal_p128_z256_drop0.2_cn0_L100_M0.1_C0_mar0.0_no_multifine/30_net_reid_encoder.pth
	# --resume /mnt/lustre/geyixiao/ECCV2018/pytorch-CycleGAN-and-pix2pix/checkpoints/nips/siamese-loss/market1501/stage2_cat_normal_p128_z256_drop0.2_cn0_L100_M0.0001_C10_mar90.0_earser/50_net_reid_encoder.pth

	# --resume /mnt/lustre/geyixiao/ECCV2018/model_zoo/siamese-loss/cuhk03/model_best.pth.tar
	# --data-dir /mnt/lustre/geyixiao/ECCV2018/open-reid-pair \

# srun -p Test --gres=gpu:1 -n1 --ntasks-per-node=1 \
# 	python examples/test.py --evaluate -d market1501 \
# 	--resume /mnt/lustre/geyixiao/ECCV2018/pytorch-CycleGAN-and-pix2pix/checkpoints/siamese-loss/market1501/stage2_cat_normal_p128_z256_drop0.2_cn0_L100_M0.1_C10_mar0.0_earser_multifine/15_net_reid_encoder.pth
	# --resume /mnt/lustre/geyixiao/ECCV2018/model_zoo/siamese-loss/market1501/model_best.pth.tar
