from __future__ import print_function, absolute_import
import time
import random
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss, SupervisedClusteringLoss
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterion, alpha=0, grp_num=1, num_classes=0, num_instances=4):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.num_classes = num_classes
        self.num_instances = num_instances
        self.alpha = alpha
        self.grp_num = grp_num

    def train(self, epoch, data_loader, optimizer, base_lr, warm_up=False, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        
        warm_up_ep = 20
        warm_iters = float(len(data_loader) * warm_up_ep)

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)

            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))
            if warm_up: 
                if epoch <= (warm_up_ep):
                    lr = (base_lr / warm_iters) + (epoch*len(data_loader) +(i+1))*(base_lr / warm_iters)
                    print(lr)
                    for g in optimizer.param_groups:
                        g['lr'] = lr * g.get('lr_mult', 1)
                else:
                    lr = base_lr
                    for g in optimizer.param_groups:
                        g['lr'] = lr * g.get('lr_mult', 1)

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
        num_classes = self.num_classes
        Y = torch.zeros(targets.size(0), num_classes)
        for i in range(targets.size(0)):
            Y[i][targets[i].cpu().data] = 1.0
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
        elif isinstance(self.criterion, SupervisedClusteringLoss):
            loss = self.criterion(outputs, Y)
            prec = 0.0
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec


class RandomWalkTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        #targets label generation
        pairwise_targets = Variable(torch.zeros(targets.size(0),\
                                int(targets.size(0) - targets.size(0) / self.num_instances)).cuda())
        targets = targets.view(int(targets.size(0) / self.num_instances), -1)
        probe_targets = targets[:,0]
        gallery_targets = targets[:, 1:self.num_instances].contiguous().view(-1)
        targets = targets.view(-1)
        for i in range(int(targets.size(0) / self.num_instances)):
            pairwise_targets[i] = (probe_targets[i] == gallery_targets).long()
        for j in range(int(targets.size(0) / self.num_instances), targets.size(0)):
            pairwise_targets[j] = (gallery_targets[j - int(targets.size(0) / self.num_instances)] == gallery_targets).long()
        pairwise_targets = pairwise_targets.view(-1).long()
        outputs = self.model(*inputs)

        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, pairwise_targets)
            prec, = accuracy(outputs.data, pairwise_targets.data)
            prec = prec[0]
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec


class RandomWalkGrpTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        #targets label generation
        pairwise_targets = Variable(torch.zeros(targets.size(0),\
                                int(targets.size(0) - targets.size(0) / self.num_instances)).cuda())
        targets = targets.view(int(targets.size(0) / self.num_instances), -1)
        probe_targets = targets[:,0]
        gallery_targets = targets[:, 1:self.num_instances].contiguous().view(-1)
        targets = targets.view(-1)
        for i in range(int(targets.size(0) / self.num_instances)):
            pairwise_targets[i] = (probe_targets[i] == gallery_targets).long()
        for j in range(int(targets.size(0) / self.num_instances), targets.size(0)):
            pairwise_targets[j] = (gallery_targets[j - int(targets.size(0) / self.num_instances)] == gallery_targets).long()
        pairwise_targets = pairwise_targets.view(-1).long()
        pairwise_targets = pairwise_targets.repeat(self.grp_num)
        outputs = self.model(*inputs)

        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, pairwise_targets)
            prec, = accuracy(outputs.data, pairwise_targets.data)
            prec = prec[0]
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

class AdvTrainer(object):
    def __init__(self, base_model, noise_embed, criterion,criterion_adv, alpha=0, grp_num=1, num_classes=0, num_instances=4):
        super(AdvTrainer, self).__init__()
        self.base_model = base_model
        self.noise_embed = noise_embed
        self.criterion = criterion
        self.criterion_adv = criterion_adv
        self.num_classes = num_classes
        self.num_instances = num_instances
        self.alpha = alpha
        self.grp_num = grp_num

    def train(self, epoch, data_loader, base_optimizer, noise_optimizer, base_lr, print_freq=1):
        self.base_model.train()
        self.noise_embed.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_base = AverageMeter()
        losses_noise = AverageMeter()
        losses_adv = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets, pids = self._parse_data(inputs)

            z1,z2,loss_b,loss_z, prec1 = self._base_forward(inputs, targets,pids)
            loss_base = loss_b + loss_z*0.1
            losses_base.update(loss_b.data[0], targets.size(0))
            losses_noise.update(loss_z.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))
            base_optimizer.zero_grad()
            loss_base.backward()
            base_optimizer.step()

            loss_adv = self._embed_forward(z1,z2,targets,pids)
            losses_adv.update(loss_adv.data[0], targets.size(0))
            loss_adv = loss_adv*0.1
            noise_optimizer.zero_grad()
            loss_adv.backward()
            noise_optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_base {:.3f} ({:.3f})\t'
                      'Loss_noise {:.3f} ({:.3f})\t'
                      'Loss_adv {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_base.val, losses_base.avg,
                              losses_noise.val, losses_noise.avg,
                              losses_adv.val, losses_adv.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        (imgs1, _, pids1, _), (imgs2, _, pids2, _) = inputs
        inputs = [Variable(imgs1), Variable(imgs2)]
        pids = Variable(torch.cat((pids1,pids2),dim=0).long().cuda())
        targets = Variable((pids1 == pids2).long().cuda())
        return inputs, targets, pids

    def _base_forward(self, inputs, targets, pids):
        z1,z2,outputs = self.base_model(*inputs)
        outputz = self.noise_embed(z1,z2)
        # outputz = F.sigmoid(*outputz)

        loss_b = self.criterion(outputs, targets)
        # targetz = Variable(torch.ones(targets.size()).cuda()*0.5)
        # loss_z = self.criterion_adv(outputz.view(-1), targetz)
        loss_z = outputz.mean()
        prec1, = accuracy(outputs.data, targets.data)
        return z1,z2,loss_b,loss_z, prec1[0]

    def _embed_forward(self, z1,z2, targets, pids):
        outputz = self.noise_embed(z1.detach(),z2.detach())
        # outputz = F.sigmoid(outputz)
        # delta = random.uniform(0.0,0.3)
        # targets = torch.abs(targets.float()-delta)
        # loss = self.criterion_adv(outputz.view(-1), targets)
        loss = (-outputz.gather(1, pids.view(-1,1))).mean()
        return loss