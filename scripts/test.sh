#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python examples/pairwise_loss.py --combine-trainval --evaluate -d market1501 --features 2048 \
	--resume /home/yxge/ECCV2018/model_zoo/market_lr01_bs224_from_imgnet/model_best.pth.tar
