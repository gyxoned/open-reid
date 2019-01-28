from __future__ import print_function, absolute_import
import time
from itertools import cycle

import torch
from torch.autograd import Variable
from torch.nn import functional as F

from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)

            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec


class CrossDomainTrainer(object):
    def __init__(self, model, netC, netD, criterion):
        super(CrossDomainTrainer, self).__init__()
        self.model = model
        self.netC = netC
        self.netD = netD
        self.criterion = criterion

    def train(self, epoch, data_loader_source, data_loader_target, optimizer, optimizer_D, print_freq=1):
        self.model.train()
        self.netC.train()
        self.netD.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_cls = AverageMeter()
        losses_dom = AverageMeter()
        losses_D = AverageMeter()
        precisions = AverageMeter()

        target_iter = iter(cycle(data_loader_target))

        end = time.time()
        for i, inputs in enumerate(data_loader_source):
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(inputs)
            t_inputs, _ = self._parse_data(next(target_iter))

            s_features = self.model(s_inputs)
            t_features = self.model(t_inputs)
            features = torch.cat((s_features, t_features))

            s_cls_out = self.netC(s_features)

            # backward netD #
            s_dom_targets = Variable(torch.zeros(s_features.size(0)).cuda())
            t_dom_targets = Variable(torch.ones(t_features.size(0)).cuda())
            dom_targets = torch.cat((s_dom_targets,t_dom_targets)).long()
            dom_out = self.netD(features.detach())
            loss_D = F.cross_entropy(dom_out, dom_targets)

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # backward main #
            loss_cls, prec1 = self._forward(s_cls_out, targets)
            loss_dom = torch.mean(F.log_softmax(self.netD(features),1))
            loss = loss_cls+loss_dom*0.1

            losses_cls.update(loss_cls.data[0], targets.size(0))
            losses_dom.update(loss_dom.data[0], features.size(0))
            losses_D.update(loss_D.data[0], features.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_cls {:.3f} ({:.3f})\t'
                      'Loss_dom {:.3f} ({:.3f})\t'
                      'Loss_D {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader_source),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_cls.val, losses_cls.avg,
                              losses_dom.val, losses_dom.avg,
                              losses_D.val, losses_D.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = Variable(imgs)
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, outputs, targets):
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec


class SiameseTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        (imgs1, _, pids1, _), (imgs2, _, pids2, _) = inputs
        inputs = [Variable(imgs1), Variable(imgs2)]
        targets = Variable((pids1 == pids2).long().cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        loss = self.criterion(outputs, targets)
        prec1, = accuracy(outputs.data, targets.data)
        return loss, prec1[0]

class InferenceBN(object):
    def __init__(self, model):
        super(InferenceBN, self).__init__()
        self.model = model

    def train(self, data_loader, print_freq=10):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, _ = self._parse_data(inputs)
            outputs = self.model(*inputs)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets