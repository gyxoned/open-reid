from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.trainers import Trainer, AdaptTrainer
from reid.evaluators import Evaluator, AdaptEvaluator
from reid.utils.data import transforms as T
from reid.utils.data.sampler import RandomMultipleGallerySampler
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint


def get_data(name, split_id, data_dir, height, width, batch_size, workers, num_instances,
             combine_trainval):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomSizedEarser(),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler_type = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler_type = None
    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=sampler_type,
        shuffle=not rmgs_flag, pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, val_loader, test_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    else:
        log_dir = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)
    dataset_source, num_classes, train_loader_source, val_loader_source, _ = \
        get_data(args.dataset_source, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers, args.num_instances,
                 args.combine_trainval)

    dataset_target, _, train_loader_target, _, test_loader_target = \
        get_data(args.dataset_target, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers, 0, True)

    # Create model
    model = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=num_classes)
    # Load from checkpoint
    start_epoch = best_mAP = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['best_mAP']
        print("=> Start epoch {}  best mAP {:.1%}"
              .format(start_epoch, best_mAP))
    model = nn.DataParallel(model).cuda()

    # Distance metric
    #metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = AdaptEvaluator(model)
    # evaluator = Evaluator_ABN(model, dataset=args.dataset)
    if args.evaluate:
        #metric.train(model, train_loader)
        #print("Validation:")
        # evaluator.evaluate(val_loader, dataset_ul.val, dataset_ul.val)
        # del model, evaluator
        # model_target = models.create(args.arch, pretrained=False, num_features=args.features, dropout=args.dropout, num_classes=num_classes), adaptation=True)
        # model_target.load_state_dict(checkpoint['state_dict'])
        # model_target = nn.DataParallel(model_target).cuda()
        # evaluator = Evaluator(model_target)
        print("Test:")
        evaluator.evaluate(test_loader_target, train_loader_target, dataset_target.query, dataset_target.gallery)
        return

    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()

    # Optimizer
    if hasattr(model.module, 'base'):
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.module.base.parameters(), 'lr_mult': 1.0},
            {'params': new_params, 'lr_mult': 10.0}]
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
   
    #optimizer = torch.optim.Adam(param_groups, lr=args.lr,
    #                             weight_decay=args.weight_decay)
    # Trainer
    trainer = AdaptTrainer(model, criterion)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = 60 if args.arch == 'inception' else args.ss
        # For warm up learning rate
        lr = args.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader_source, train_loader_target, optimizer)
        if epoch < args.start_save:
            continue
        _, mAP = evaluator.evaluate(val_loader_source, train_loader_target, dataset_source.val, dataset_source.val)

        is_best = mAP > best_mAP
        best_mAP = max(mAP, best_mAP)
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_mAP': best_mAP,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  mAP: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, mAP, best_mAP, ' *' if is_best else ''))

    # Final test
    del model, evaluator
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model_target = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=num_classes, adaptation=False)
    model_target.load_state_dict(checkpoint['state_dict'])
    model_target = nn.DataParallel(model_target).cuda()
    evaluator = Evaluator(model_target)
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-dt', '--dataset-target', type=str, default='dukemtmc',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--sphere', action='store_true',
                        help = "use sphere")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--ss', type=int, default=20)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())
