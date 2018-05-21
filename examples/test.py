from __future__ import print_function
import argparse
import os.path as osp
import operator

import numpy as np
import sys
sys.path.append('./')
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from reid import datasets
from reid.datasets import create
# from reid.mining import mine_hard_pairs
from reid.models import ResNet
from reid.models.embedding import EltwiseSubEmbed
    # KronEmbed, HourGlassEltwiseSubEmbed, \
    #  HourGlassEltwiseKronEmbed, HourGlassEltwiseKronEmbedSelfAtt
from reid.models.multi_branch import SiameseNet\
    # , SiameseHourGlassNet
from reid.trainers import SiameseTrainer\
    # , SiameseHourGlassTrainer
from reid.evaluators import CascadeEvaluator
from reid.utils.data import transforms
from reid.utils.data.sampler import RandomPairSampler, SubsetRandomSampler, RandomMultipleGallerySampler
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, \
    copy_state_dict

from reid.utils.serialization import write_json, read_json
from reid.feature_extraction import extract_cnn_feature
from collections import OrderedDict
import pdb
import pickle
import json

pid_select=[260, 1, 227, 228, 94, 183, 155, 727, 270, 92, 156, 521, 133, 710, 770, 812, 137, 731]

def _pluck0(identities, indices, relabel=False):
    ret = {}
    for index, pid in enumerate(indices):
        img_num = 0
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                img_num += 1
        ret[pid] = img_num
    return ret


def _pluck(identities, indices):
    ret = []
    for index, pid in enumerate(indices):
        # img_num = 0
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                name = osp.splitext(fname)[0]
                x, y, _ = map(int, name.split('_'))
                assert pid == x and camid == y
                ret.append((fname, pid, camid))
                # img_num += 1
            #     if img_num==50:
            #         break
            # if img_num==50:
            #     break
        # assert (img_num==50)
    return ret

def get_data(dataset_name, split_id, data_dir, batch_size, workers, combine_trainval, np_ratio):
    root = osp.join(data_dir, dataset_name)

    dataset = create(dataset_name, root,
                          split_id=split_id, num_val=100, download=True)

    indices = list(set(dataset.split['gallery']).intersection(set(dataset.split['query'])))
    # ret = _pluck0(dataset.meta['identities'], indices)
    # sort_ret= sorted(ret.items(), key=operator.itemgetter(1), reverse=True)

    #
    # for i in range(20):
    #     print (str(sort_ret[i][0])+':'+str(sort_ret[i][1]))
    testset = _pluck(dataset.meta['identities'], indices)

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    # train_set = dataset.trainval if combine_trainval else dataset.train
    #
    # train_loader = DataLoader(
    #     Preprocessor(train_set, root=dataset.images_dir,
    #                  transform=transforms.Compose([
    #                      transforms.RectScale(256, 128),
    #                      transforms.RandomSizedEarser(),
    #                      transforms.RandomHorizontalFlip(),
    #                      transforms.ToTensor(),
    #                      normalizer,
    #                  ])),
    #     sampler=RandomPairSampler(train_set, neg_pos_ratio=np_ratio),
    #     # sampler=RandomMultipleGallerySampler(train_set),
    #     batch_size=batch_size, num_workers=workers, pin_memory=False)
    #
    # val_loader = DataLoader(
    #     Preprocessor(dataset.val, root=dataset.images_dir,
    #                  transform=transforms.Compose([
    #                      transforms.RectScale(256, 128),
    #                      transforms.ToTensor(),
    #                      normalizer,
    #                  ])),
    #     batch_size=batch_size, num_workers=workers,
    #     shuffle=False, pin_memory=False)

    test_loader = DataLoader(
        Preprocessor(testset,
                     root=dataset.images_dir,
                     transform=transforms.Compose([
                         transforms.RectScale(256, 128),
                         transforms.ToTensor(),
                         normalizer,
                     ])),
        batch_size=batch_size, num_workers=0,
        shuffle=False, pin_memory=False)

    return dataset, test_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    dataset, test_loader = get_data(args.dataset, args.split, args.data_dir,
             args.batch_size, args.workers, args.combine_trainval, args.np_ratio)

    # features = read_json(osp.join(dataset.root, 'features.json'))
    # labels = read_json(osp.join(dataset.root, 'labels.json'))
    # import pdb; pdb.set_trace()

    base_model = ResNet(args.depth, num_classes=0,
                        cut_at_pooling=True,
                        num_features=args.features, dropout=args.dropout)
    embed_model = EltwiseSubEmbed(use_batch_norm=True,
                                  use_classifier=True,
                                  num_features=args.features, num_classes=2)

    model = SiameseNet(base_model, embed_model)
    model = torch.nn.DataParallel(model.cuda())

    checkpoint = load_checkpoint(args.resume)
    if args.resume.endswith('.tar'):
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))
    else:
        # for key in checkpoint.keys():
        #     checkpoint[key[7:]]=checkpoint.pop(key)
        model.load_state_dict(checkpoint)

    model.eval()
    features = {}
    labels = {}

    for i, (imgs, fnames, pids, _) in enumerate(test_loader):
        outputs = extract_cnn_feature(torch.nn.DataParallel(base_model.cuda()), imgs)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output.numpy().tolist()
            labels[fname] = pid
    write_json(features, osp.join(dataset.root,'visual_full','LMC','features.json'))
    write_json(labels, osp.join(dataset.root,'visual_full','LMC','labels.json'))
    print ('finish')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Training Inception Siamese Model")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--hard-examples', action='store_true')
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, ")
    # model
    parser.add_argument('--depth', type=int, default=50,
                        choices=[18, 34, 50, 101, 152])
    parser.add_argument('--features', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--embedding', type=str, default='sub',
                        choices=['kron', 'sub', 'hgkron', 'hgsub', 'hgkronsa'])
    # loss
    parser.add_argument('--loss', type=str, default='xentropy',
                        choices=['xentropy'])
    parser.add_argument('--margin', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--np-ratio', type=int, default=3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--ss', type=int, default=40)
    # training configs
    parser.add_argument('--retrain', type=str, default='', metavar='PATH')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())
