#!/bin/bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
	python examples/adv_loss.py --combine-trainval --logs-dir logs/advloss-soft/market-resnet50 \
		--lr 0.01 -b 256 -j 8 --features 1024 --noise 256
