from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp
import random
import numpy as np
import sys

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.sampler import RandomMultipleGallerySampler, DistributedRandomMultipleGallerySampler
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint


start_epoch = best_mAP = 0


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

    assert(num_instances > 0)
    sampler_type = DistributedRandomMultipleGallerySampler(train_set, num_instances)

    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=sampler_type, shuffle=False, pin_memory=True, drop_last=True)

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

    return dataset, num_classes, train_loader, val_loader, test_loader, sampler_type


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    # debug
    # main_worker(0, ngpus_per_node, args)

# def main_worker(ngpus_per_node, args):
def main_worker(gpu, ngpus_per_node, args):
    global start_epoch, best_mAP
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node

    print("Use GPU: {} for training".format(args.gpu))
    # if args.distributed:
    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    if args.rank % ngpus_per_node == 0:
        if not args.evaluate:
            sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
        else:
            log_dir = osp.dirname(args.resume)
            sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
        print("==========\nArgs:{}\n==========".format(args))

    batch_size = int(args.batch_size / ngpus_per_node)
    workers = int(args.workers / ngpus_per_node)
    print("batch size is {}, worker is {}".format(batch_size, workers))
    # Create data loaders
    #assert args.num_instances > 1, "num_instances should be greater than 1"
    #assert args.batch_size % args.num_instances == 0, \
    #    'num_instances should divide batch_size'
    dataset, num_classes, train_loader, val_loader, test_loader, train_sampler = \
        get_data(args.dataset, args.split, args.data_dir, args.height,
                 args.width, batch_size, workers, args.num_instances,
                 args.combine_trainval)

    # Create model
    model = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=num_classes, pretrained=True)
    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['best_mAP']
        print("=> Start epoch {}  best mAP {:.1%}"
              .format(start_epoch, best_mAP))

    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)

    # Distance metric
    # metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = Evaluator(model, dataset=args.dataset)
    if args.evaluate:
        # metric.train(model, train_loader)
        # print("Validation:")
        # evaluator.evaluate(val_loader, dataset.val, dataset.val)
        print("Test:")
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True, args=args)
        return

    # Criterion
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

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
   
    # Trainer
    trainer = Trainer(model, criterion)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = args.step_size
        lr = args.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer, args=args)
        if epoch < args.start_save:
            continue
        mAP = evaluator.evaluate(val_loader, dataset.val, dataset.val, args=args)

        is_best = mAP > best_mAP
        best_mAP = max(mAP, best_mAP)

        if args.rank % ngpus_per_node == 0:
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    # metric.train(model, train_loader)
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True, args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default=128,
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
    # optimizer
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    # parser.add_argument('--seed', type=int, default=1)
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
    # distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://10.1.72.207:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    # parser.add_argument('--gpu', default=None, type=int,
    #                     help='GPU id to use.')
    # parser.add_argument('-mp', '--multiprocessing-distributed', action='store_true',
    #                     help='Use multi-processing distributed training to launch '
    #                          'N processes per node, which has N GPUs. This is the '
    #                          'fastest way to use PyTorch for either single node or '
    #                          'multi node data parallel training')
    main()
