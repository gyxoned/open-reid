from __future__ import print_function
import argparse
import os.path as osp

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
from reid.utils.data.sampler import RandomPairSampler, SubsetRandomSampler
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, \
    copy_state_dict
import pdb
import pickle



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
        sampler=RandomPairSampler(train_set, neg_pos_ratio=np_ratio),
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
    torch.cuda.manual_seed_all(args.seed)

    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    dataset, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir,
                 args.batch_size, args.workers, args.combine_trainval, args.np_ratio)

    # Create models
    if args.embedding == 'kron':
        base_model = ResNet(args.depth, cut_at_pooling=True)
        embed_model = KronEmbed(8, 4, args.features, 2)
    elif args.embedding == 'hgkronsa':
        base_model = ResNet(args.depth, num_classes=0,
                            cut_at_pooling=True,
                            num_features=args.features, dropout=args.dropout)
        embed_model = HourGlassEltwiseKronEmbedSelfAtt(use_batch_norm=True,
                                      use_classifier=True,
                                      num_features=args.features, num_classes=2)
    elif args.embedding == 'hgkron':
        base_model = ResNet(args.depth, num_classes=0,
                            cut_at_pooling=True,
                            num_features=args.features, dropout=args.dropout)
        embed_model = HourGlassEltwiseKronEmbed(use_batch_norm=True,
                                      use_classifier=True,
                                      num_features=args.features, num_classes=2)
    elif args.embedding == 'hgsub':
        base_model = ResNet(args.depth, num_classes=0,
                            cut_at_pooling=True,
                            num_features=args.features, dropout=args.dropout)
        embed_model = HourGlassEltwiseSubEmbed(use_batch_norm=True,
                                      use_classifier=True,
                                      num_features=args.features, num_classes=2)
    else:
        base_model = ResNet(args.depth, num_classes=0,
                            cut_at_pooling=True,
                            num_features=args.features, dropout=args.dropout)
        embed_model = EltwiseSubEmbed(use_batch_norm=True,
                                      use_classifier=True,
                                      num_features=args.features, num_classes=2)

    if (args.embedding == 'hgkron') or (args.embedding == 'hgsub') or (args.embedding == 'hgkronsa'):
        model = SiameseHourGlassNet(base_model, embed_model)
    else:
        model = SiameseNet(base_model, embed_model)
    model = torch.nn.DataParallel(model.cuda())

    if args.retrain:
        checkpoint = load_checkpoint(args.retrain)
        print("loading base part")
        copy_state_dict(checkpoint['state_dict'], base_model, strip='module.base_model.')
        print("loading embed part")
        copy_state_dict(checkpoint['state_dict'], embed_model, strip='module.embed_model.')

    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        best_mAP = checkpoint['best_mAP']
        print("=> start epoch {}  best top1 {:.1%}"
              .format(args.start_epoch, best_top1))
    else:
        best_mAP = 0
        best_top1 = 0

    # Evaluator
    evaluator = CascadeEvaluator(
        torch.nn.DataParallel(base_model).cuda(),
        embed_model,
        embed_dist_fn=lambda x: F.softmax(Variable(x)).data[:, 0])
    if args.evaluate:
        # pdb.set_trace()
        # #print("Validation:")
        # #evaluator.evaluate(val_loader, dataset.val, dataset.val)
        print("Test:")
    # with open('market1501query', 'wb') as fp:
    #     pickle.dump(dataset.query, fp)
    # with open('market1501gallery', 'wb') as fp:
    #             pickle.dump(dataset.gallery, fp)
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, rerank_topk=100, dataset=args.dataset)
        return

    # if args.hard_examples:
    #     # Use sequential train set loader
    #     data_loader = DataLoader(
    #         Preprocessor(dataset.trainval, root=dataset.images_dir,
    #                      transform=val_loader.dataset.transform),
    #         batch_size=args.batch_size, num_workers=args.workers,
    #         shuffle=False, pin_memory=False)
    #     # Mine hard triplet examples, index of [(anchor, pos, neg), ...]
    #     pairs = mine_hard_pairs(torch.nn.DataParallel(base_model).cuda(),
    #                             data_loader, margin=args.margin)
    #     print("Mined {} hard example triplets".format(len(pairs)))
    #     # Build a hard examples loader
    #     train_loader.sampler = SubsetRandomSampler(pairs)

    # Criterion
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    #
    optimizer = torch.optim.SGD([
                               {'params': model.module.base_model.parameters()},
                               {'params': model.module.embed_model.parameters(), 'lr': args.lr*10}
                               ], args.lr,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay)
#    optimizer = torch.optim.SGD([
#                                {'params': model.module.base_model.base.layer4.parameters()},
#                                {'params': model.module.embed_model.parameters(), 'lr': args.lr*10}
#                                ], args.lr,
#                                momentum=args.momentum,
#                                weight_decay=args.weight_decay)



    # Trainer
    trainer = SiameseTrainer(model, criterion)
    #trainer = SiameseHourGlassTrainer(model, criterion)

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
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=2)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--hard-examples', action='store_true')
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, ")
    # model
    parser.add_argument('--depth', type=int, default=50,
                        choices=[18, 34, 50, 101, 152])
    parser.add_argument('--features', type=int, default=256)
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
