from __future__ import print_function, absolute_import
import os, sys
import argparse
import os.path as osp
import copy
import numpy as np
from sklearn.cluster import AgglomerativeClustering

import torch
from torch import nn
from torch.backends import cudnn
from torch.nn import init
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.trainers import Trainer, InferenceBN, TeacherStudentTrainer
from reid.evaluators import Evaluator, Evaluator_ABN, extract_features
from reid.utils.data import transforms as T
from reid.utils.data.sampler import RandomMultipleGallerySampler
from reid.utils.data.preprocessor import Preprocessor, Preprocessor_double
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint


def get_data(name, split_id, data_dir, combine_trainval):
# , height, width, batch_size, workers, num_instances, combine_trainval):
    # root = osp.join(data_dir, name)
    root = data_dir

    dataset = datasets.create(name, root, split_id=split_id)

    # normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])

    # train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    # train_transformer = T.Compose([
    #     T.RandomSizedRectCrop(height, width),
    #     T.RandomSizedEarser(),
    #     T.RandomHorizontalFlip(),
    #     T.ToTensor(),
    #     normalizer,
    # ])

    # test_transformer = T.Compose([
    #     T.RectScale(height, width),
    #     T.ToTensor(),
    #     normalizer,
    # ])

    # rmgs_flag = num_instances > 0
    # if rmgs_flag:
    #     sampler_type = RandomMultipleGallerySampler(train_set, num_instances)
    # else:
    #     sampler_type = None
    # train_loader = DataLoader(
    #     Preprocessor(train_set, root=dataset.images_dir,
    #                  transform=train_transformer),
    #     batch_size=batch_size, num_workers=workers,
    #     sampler=sampler_type,
    #     shuffle=not rmgs_flag, pin_memory=True, drop_last=True)

    # val_loader = DataLoader(
    #     Preprocessor(dataset.val, root=dataset.images_dir,
    #                  transform=test_transformer),
    #     batch_size=batch_size, num_workers=workers,
    #     shuffle=False, pin_memory=True)

    # test_loader = DataLoader(
    #     Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
    #                  root=dataset.images_dir, transform=test_transformer),
    #     batch_size=batch_size, num_workers=workers,
    #     shuffle=False, pin_memory=True)

    return dataset, num_classes
    # , train_loader, val_loader, test_loader

def get_train_loader(dataset, split_classes, height, width, batch_size, workers, num_instances, combine_trainval=True):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomSizedEarser(),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    train_set = dataset.trainval if combine_trainval else dataset.train

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler_type = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler_type = None

    train_loader = DataLoader(
        Preprocessor_double(train_set, split_classes, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=sampler_type,
        shuffle=not rmgs_flag, pin_memory=True, drop_last=True)

    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_model(args, num_classes):
    student_model = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=0)
    teacher_model = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=0)
    netC = nn.Linear(args.features, num_classes, bias=True)
    netD = nn.Linear(args.features, 2, bias=True)
    # init.normal(netC.weight, std=0.001)
    # init.constant(netC.bias, 0)
    # init.normal(netD.weight, std=0.001)
    # init.constant(netD.bias, 0)
    initial_weights = load_checkpoint(args.init)
    student_model.load_state_dict(initial_weights['model_state_dict'])
    teacher_model.load_state_dict(initial_weights['model_state_dict'])
    netC.load_state_dict(initial_weights['netC_state_dict'])
    netD.load_state_dict(initial_weights['netD_state_dict'])

    return student_model, teacher_model, netC, netD


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
    # assert args.num_instances > 1, "num_instances should be greater than 1"
    # assert args.batch_size % args.num_instances == 0, \
    #     'num_instances should divide batch_size'
    assert args.init, 'model should begin with pretrained weights'
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)

    dataset_source, num_classes = get_data(args.dataset_source, args.split, args.data_dir, args.combine_trainval)
        # , args.height, args.width, args.batch_size, args.workers, args.num_instances,
        #          args.combine_trainval)
    test_loader_source = get_test_loader(dataset_source, args.height, args.width, args.batch_size, args.workers)
    dataset_target, _ = get_data(args.dataset_target, args.split, args.data_dir, args.combine_trainval)
        # , args.height,
        #          args.width, args.batch_size, args.workers, args.num_instances, True)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)

    # Create model
    student_model, teacher_model, netC, netD = create_model(args, num_classes)

    # Load from checkpoint
    # start_epoch = best_mAP = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        num_cluster = checkpoint['num_cluster']
        netC = nn.Linear(args.features, num_classes+num_cluster, bias=True)

        student_model.load_state_dict(checkpoint['model_state_dict'])
        netC.load_state_dict(checkpoint['netC_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        # start_epoch = checkpoint['epoch']
        # best_mAP = checkpoint['best_mAP']
        # print("=> Start epoch {}  best mAP {:.1%}"
        #       .format(start_epoch, best_mAP))

    student_model = nn.DataParallel(student_model).cuda()
    teacher_model = nn.DataParallel(teacher_model).cuda()
    netC = nn.DataParallel(netC).cuda()
    netD = nn.DataParallel(netD).cuda()

    # Evaluator
    evaluator = Evaluator(student_model)   
    if args.evaluate:
        print("Test source domain:")
        evaluator.evaluate(test_loader_source, dataset_source.query, dataset_source.gallery)
        print("Test target domain:")
        evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery)
        return

    # clusters = [50, 200, 300, 350]
    # epochs = [20, 30, 40, 50]

    clusters = [100]
    epochs = [50]

    for nc in range(len(clusters)):
        teacher_model.load_state_dict(student_model.state_dict())

        # TODO cluster
        cluster_loader = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers, testset=dataset_target.trainval)
        dict_f, _ = extract_features(teacher_model, cluster_loader)
        f = torch.stack(list(dict_f.values())).numpy()

        print('\n Clustering into {} classes \n'.format(clusters[nc]))
        c_predict = AgglomerativeClustering(n_clusters=clusters[nc]).fit_predict(f)
        c_predict = c_predict + num_classes
        for i in range(len(dataset_target.trainval)): 
            dataset_target.trainval[i] = list(dataset_target.trainval[i])
            dataset_target.trainval[i][1] = int(c_predict[i])
            dataset_target.trainval[i] = tuple(dataset_target.trainval[i])
        dataset_target.val = dataset_target.trainval[-len(dataset_target.val):]

        # TODO update netC
        new_weight = torch.FloatTensor(num_classes+clusters[nc], args.features).cuda()
        new_bias = torch.FloatTensor(num_classes+clusters[nc]).cuda()
        init.normal(new_weight, std=0.001)
        init.constant(new_bias, 0)
        new_weight[:num_classes] = netC.state_dict()['module.weight'][:num_classes]
        new_bias[:num_classes] = netC.state_dict()['module.bias'][:num_classes]
        netC = nn.Linear(args.features, num_classes+clusters[nc], bias=True)
        netC.state_dict()['weight'] = new_weight
        netC.state_dict()['bias'] = new_bias
        netC = nn.DataParallel(netC).cuda()

        # TODO data loader
        train_loader_source = get_train_loader(dataset_source, num_classes, args.height, args.width, 
            args.batch_size, args.workers, args.num_instances, args.combine_trainval)
        train_loader_target = get_train_loader(dataset_target, num_classes, args.height, args.width, 
            args.batch_size, args.workers, args.num_instances, args.combine_trainval)
        valset = list(set(dataset_source.val) | set(dataset_target.val))
        val_loader = get_test_loader(dataset_source, args.height, args.width, 
            args.batch_size, args.workers, testset=valset)

        # Optimizer
        lr = args.lr * (0.1 ** nc)
        base_param_ids = set(map(id, student_model.module.base.parameters()))
        new_params = [p for p in student_model.parameters() if id(p) not in base_param_ids]
        param_groups = [{'params': student_model.module.base.parameters(), 'lr_mult': 1.0},
                        {'params': new_params, 'lr_mult': 10.0},
                        {'params': netC.parameters(), 'lr_mult': 100.0}]

        optimizer = torch.optim.SGD(param_groups, lr=lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
        optimizer_D = torch.optim.SGD(netD.parameters(), lr=lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)

        # Trainer
        trainer = TeacherStudentTrainer(student_model, teacher_model, netC, netD)

        # Schedule learning rate
        def adjust_lr(epoch):
            step_size = args.ss
            new_lr = lr * (0.1 ** (epoch // step_size))
            for g in optimizer.param_groups:
                g['lr'] = new_lr * g.get('lr_mult', 1)

        best_mAP = 0
        # Start training
        for epoch in range(0, epochs[nc]):
            adjust_lr(epoch)
            trainer.train(clusters[nc], epoch, train_loader_source, train_loader_target, optimizer, optimizer_D)
            # if epoch < args.start_save:
            #     continue
            _, mAP = evaluator.evaluate(val_loader, valset, valset)

            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'model_state_dict': student_model.module.state_dict(),
                'netC_state_dict': netC.module.state_dict(),
                'netD_state_dict': netD.module.state_dict(),
                'cluster': clusters[nc],
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'cls'+str(clusters[nc])+'_checkpoint.pth.tar'))

            print('\n * [Cluster {}] Finished epoch {:3d}  mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(clusters[nc], epoch, mAP, best_mAP, ' *' if is_best else ''))

        checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
        student_model.module.load_state_dict(checkpoint['model_state_dict'])
        os.rename(osp.join(args.logs_dir, 'model_best.pth.tar'), osp.join(args.logs_dir, 'cls'+str(clusters[nc])+'_model_best.pth.tar'))

        # Final test
        print('\n * [Cluster {}] Test source domain:'.format(clusters[nc]))
        evaluator.evaluate(test_loader_source, dataset_source.query, dataset_source.gallery)
        print('\n * [Cluster {}] Test target domain:'.format(clusters[nc]))
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
    parser.add_argument('--init', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=70)
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
