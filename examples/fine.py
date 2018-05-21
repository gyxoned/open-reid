from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
sys.path.append('./')
import torch
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable


from reid import datasets
from reid import models
from reid.datasets import create
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss
from reid.trainers import Trainer, RandomWalkTrainer
from reid.evaluators import Evaluator, CascadeEvaluator
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.models.embedding import RandomWalkEmbed
from reid.models.multi_branch import RandomWalkNet
from reid.models import ResNet
from reid.models.embedding import EltwiseSubEmbed
from reid.models.multi_branch import SiameseNet
from reid.utils.data import transforms
from reid.utils.data.sampler import RandomPairSampler, SubsetRandomSampler, RandomMultipleGallerySampler
import pdb

def get_data(dataset_name, split_id, data_dir, batch_size, workers, combine_trainval, np_ratio):
    root = osp.join(data_dir, dataset_name)

    dataset = create(dataset_name, root,
                          split_id=split_id, num_val=100, download=True)

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train

    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=transforms.Compose([
                         transforms.RectScale(256, 128),
                         transforms.RandomSizedEarser(),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         normalizer,
                     ])),
        # sampler=RandomPairSampler(train_set, neg_pos_ratio=np_ratio),
        sampler=RandomMultipleGallerySampler(train_set),
        batch_size=batch_size, num_workers=workers, pin_memory=False)

    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=transforms.Compose([
                         transforms.RectScale(256, 128),
                         transforms.ToTensor(),
                         normalizer,
                     ])),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir,
                     transform=transforms.Compose([
                         transforms.RectScale(256, 128),
                         transforms.ToTensor(),
                         normalizer,
                     ])),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    return dataset, train_loader, val_loader, test_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    # assert args.num_instances > 1, "num_instances should be greater than 1"
    # assert args.batch_size % args.num_instances == 0, \
    #     'num_instances should divide batch_size'
    # if args.height is None or args.width is None:
    #     args.height, args.width = (144, 56) if args.arch == 'inception' else \
    #                               (256, 128)
    dataset, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir,
                 args.batch_size, args.workers, args.combine_trainval, args.np_ratio)

    # Create model
    # Hacking here to let the classifier be the last feature embedding layer
    # Net structure: avgpool -> FC(1024) -> FC(args.features)
    # base_model = models.create(args.arch, num_features=1024, cut_at_pooling=True,
    #                       dropout=args.dropout, num_classes=args.features)
    #
    # embed_model = RandomWalkEmbed(instances_num=args.num_instances,
    #                         feat_num=2048, num_classes=2)
    base_model = ResNet(args.depth, num_classes=0,
                        cut_at_pooling=True,
                        num_features=args.features, dropout=args.dropout)
    embed_model = RandomWalkEmbed(feat_num=args.features, num_classes=2, drop_ratio=0.0)

    base_model = nn.DataParallel(base_model).cuda()
    embed_model = embed_model.cuda()

    model = RandomWalkNet(base_model=base_model, embed_model=embed_model)
    # model = torch.nn.DataParallel(model.cuda())

    if args.retrain:
        print('loading base part of pretrained model...')
        checkpoint = load_checkpoint(args.retrain)
        copy_state_dict(checkpoint, base_model, strip='module.base_model.')
        # copy_state_dict(checkpoint['state_dict'], base_model, strip='module.base_model.')
        print('loading embed part of pretrained model...')
        copy_state_dict(checkpoint, embed_model, strip='module.embed_model.')
        # copy_state_dict(checkpoint['state_dict'], embed_model, strip='module.embed_model.')

    # base_model = nn.DataParallel(base_model).cuda()
    # embed_model = embed_model.cuda()
    #
    # model = RandomWalkNet(instances_num=args.num_instances,
    #                     base_model=base_model, embed_model=embed_model)

    # Distance metric
    # metric = DistanceMetric(algorithm=args.dist_metric)

        # Load from checkpoint
    start_epoch = best_top1 = 0
    best_mAP = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))

    # Evaluator
    # evaluator = CascadeEvaluator(
    #                         base_model,
    #                         embed_model,
    #                         embed_dist_fn=lambda x: F.softmax(x).data[:, 0])
    # Evaluator
    evaluator = CascadeEvaluator(
        base_model,
        embed_model,
        embed_dist_fn=lambda x: F.softmax(Variable(x)).data[:, 0])
    if args.evaluate:
        print("Test:")
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, rerank_topk=100, dataset=args.dataset)
        return

    # Criterion
    #criterion = TripletLoss(margin=args.margin).cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    # Optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
    #                              weight_decay=args.weight_decay)
    # import pdb; pdb.set_trace()
    optimizer = torch.optim.SGD([
                               {'params': model.base_model.module.parameters()},
                               {'params': model.embed_model.parameters(), 'lr': args.lr*10}
                               ], args.lr,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay)
    # Trainer
    #trainer = Trainer(model, criterion)
    trainer = RandomWalkTrainer(model, criterion)

    # Schedule learning rate
    def adjust_lr(epoch):
        lr = args.lr * (0.1 ** (epoch // args.ss))
        for g in optimizer.param_groups:
            g['lr'] = lr
        return lr

    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer, base_lr=args.lr, warm_up=False)

        top1, mAP = evaluator.evaluate(val_loader, dataset.val, dataset.val, dataset=args.dataset)

        #is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        is_best = mAP > best_mAP
        best_mAP = max(mAP, best_mAP)
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
            'best_mAP': best_mAP,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  mAP: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, mAP, best_mAP, ' *' if is_best else ''))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, dataset=args.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Training Inception Siamese Model")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
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
