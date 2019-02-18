from __future__ import print_function, absolute_import
import time
from itertools import cycle

import torch
import torch.nn as nn
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


class TeacherStudentTrainer(object):
    def __init__(self, student_model, teacher_model, netC, netD):
        super(TeacherStudentTrainer, self).__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.netC = netC
        self.netD = netD

        self.criterion_cls = nn.CrossEntropyLoss().cuda()
        self.criterion_consis = nn.MSELoss().cuda()

    def train(self, num_cluster, epoch, data_loader, optimizer, optimizer_D, print_freq=1):
        self.student_model.train()
        self.teacher_model.train() # eval()?
        self.netC.train()
        self.netD.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_cls = AverageMeter()
        losses_dom = AverageMeter()
        losses_consis = AverageMeter()
        losses_D = AverageMeter()
        precisions = AverageMeter()


        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, inputs_z, targets, dom_targets = self._parse_data(inputs)

            f_out_s = self.student_model(inputs)
            f_out_t = self.teacher_model(inputs_z).detach()

            cls_out = self.netC(f_out_s)

            # backward netD #
            dom_out = self.netD(f_out_s.detach())
            loss_D = F.cross_entropy(dom_out, dom_targets)

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # backward main #
            # TODO loss weight adjustment
            loss_cls = self.criterion_cls(cls_out, targets)
            loss_consis = self.criterion_consis(f_out_s, f_out_t)
            loss_dom = torch.mean(F.log_softmax(self.netD(f_out_s),1))
            loss = loss_cls + loss_consis + loss_dom*0.1

            prec, = accuracy(cls_out.data, targets.data)
            prec1 = prec[0]

            losses_cls.update(loss_cls.data[0], targets.size(0))
            losses_consis.update(loss_consis.data[0], f_out_s.size(0))
            losses_dom.update(loss_dom.data[0], f_out_s.size(0))
            losses_D.update(loss_D.data[0], f_out_s.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Cluster: {}\t'
                      'Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_cls {:.3f} ({:.3f})\t'
                      'Loss_con {:.3f} ({:.3f})\t'
                      'Loss_dom {:.3f} ({:.3f})\t'
                      'Loss_D {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(num_cluster, epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_cls.val, losses_cls.avg,
                              losses_consis.val, losses_consis.avg,
                              losses_dom.val, losses_dom.avg,
                              losses_D.val, losses_D.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, imgs_z, _, pids, _, domids = inputs
        inputs = Variable(imgs.cuda())
        inputs_z = Variable(imgs_z.cuda())
        targets = Variable(pids.cuda()).long()
        dom_targets = Variable(domids.cuda()).long()
        return inputs, inputs_z, targets, dom_targets


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