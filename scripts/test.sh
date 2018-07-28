CUDA_VISIBLE_DEVICES=2,3 \
python examples/siamese_loss.py -b 256 -d dukemtmc -a resnet50 --features 2048 \
	--evaluate --retrain logs/fd-gan/duke/latest_net_reid_encoder.pth