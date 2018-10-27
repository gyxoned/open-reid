from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
from torch.autograd import Variable

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature, extract_bn_responses
from .utils.meters import AverageMeter


def extract_features(model, data_loader, print_freq=1, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels

def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), dataset=None):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute all kinds of CMC scores
    if not dataset:
        cmc_configs = {
          'allshots': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=False),
          'cuhk03': dict(separate_camera_set=True,
                         single_gallery_shot=True,
                         first_match_break=False),
          'market1501': dict(separate_camera_set=False,
                             single_gallery_shot=False,
                             first_match_break=True)}
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                              query_cams, gallery_cams, **params)
                    for name, params in cmc_configs.items()}

        print('CMC Scores{:>12}{:>12}{:>12}'
            .format('allshots', 'cuhk03', 'market1501'))
        for k in cmc_topk:
          print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
                .format(k, cmc_scores['allshots'][k - 1],
                        cmc_scores['cuhk03'][k - 1],
                        cmc_scores['market1501'][k - 1]))

        # Use the allshots cmc top-1 score for validation criterion
        return cmc_scores['allshots'][0], mAP
    else:

        if (dataset == 'cuhk03'):
            cmc_configs = {
                'cuhk03': dict(separate_camera_set=True,
                                  single_gallery_shot=True,
                                  first_match_break=False),
                }
            cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                    query_cams, gallery_cams, **params)
                          for name, params in cmc_configs.items()}

            print('CMC Scores{:>12}'.format('cuhk03'))
            for k in cmc_topk:
                print('  top-{:<4}{:12.1%}'
                      .format(k,
                              cmc_scores['cuhk03'][k - 1]))
            # Use the allshots cmc top-1 score for validation criterion
            return cmc_scores['cuhk03'][0], mAP
        else:
            cmc_configs = {
                'market1501': dict(separate_camera_set=False,
                                   single_gallery_shot=False,
                                   first_match_break=True)
                        }
            cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                    query_cams, gallery_cams, **params)
                          for name, params in cmc_configs.items()}

            print('CMC Scores{:>12}'.format('market1501'))
            for k in cmc_topk:
                print('  top-{:<4}{:12.1%}'
                      .format(k,
                              cmc_scores['market1501'][k-1]))
            return cmc_scores['market1501'][0], mAP


class Evaluator(object):
    def __init__(self, model, dataset='market1501'):
        super(Evaluator, self).__init__()
        self.model = model
        self.dataset = dataset

    def evaluate(self, data_loader, query, gallery, metric=None):
        features, _ = extract_features(self.model, data_loader)
        distmat = pairwise_distance(features, query, gallery, metric=metric)
        return evaluate_all(distmat, query=query, gallery=gallery, dataset=self.dataset)

class ABN_Parameters(object):
    def __init__(self):
        self.running_mean = None
        self.running_var = None
        self.count = None

    def reset(self, dim):
        self.running_mean = torch.zeros(dim).float().cuda()
        self.running_var = torch.ones(dim).float().cuda()
        self.count = 0

    def update(self, mean, var, k=1):
        assert(k>0)
        d = mean - self.running_mean
        if self.count==0:
            self.running_mean = d
        else:
            self.running_mean = self.running_mean + d*k/self.count
        self.running_var = self.running_var*self.count/(self.count+k) + \
                           var*k/(self.count+k) + d**2*self.count*k/(self.count+k)**2
        self.count = self.count + k
    
    # def update(self, mean, var, k=1):
    #     if self.count==0:
    #         self.running_mean = mean
    #         self.running_var = var
    #     else:
    #         self.running_mean = 0.9*self.running_mean + 0.1*mean
    #         self.running_var = 0.9*self.running_var + 0.1*var

def adapt_source_bn(model, data_loader, print_freq=10):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    all_bn_modules = ['module.base.bn1', 'module.base.layer1.0.bn1', 'module.base.layer1.0.bn2', 'module.base.layer1.0.bn3', 
                    'module.base.layer1.0.downsample.1', 'module.base.layer1.1.bn1', 'module.base.layer1.1.bn2', 'module.base.layer1.1.bn3', 
                    'module.base.layer1.2.bn1', 'module.base.layer1.2.bn2', 'module.base.layer1.2.bn3', 'module.base.layer2.0.bn1', 
                    'module.base.layer2.0.bn2', 'module.base.layer2.0.bn3', 'module.base.layer2.0.downsample.1', 'module.base.layer2.1.bn1', 
                    'module.base.layer2.1.bn2', 'module.base.layer2.1.bn3', 'module.base.layer2.2.bn1', 'module.base.layer2.2.bn2', 
                    'module.base.layer2.2.bn3', 'module.base.layer2.3.bn1', 'module.base.layer2.3.bn2', 'module.base.layer2.3.bn3', 
                    'module.base.layer3.0.bn1', 'module.base.layer3.0.bn2', 'module.base.layer3.0.bn3', 'module.base.layer3.0.downsample.1', 
                    'module.base.layer3.1.bn1', 'module.base.layer3.1.bn2', 'module.base.layer3.1.bn3', 'module.base.layer3.2.bn1', 
                    'module.base.layer3.2.bn2', 'module.base.layer3.2.bn3', 'module.base.layer3.3.bn1', 'module.base.layer3.3.bn2', 
                    'module.base.layer3.3.bn3', 'module.base.layer3.4.bn1', 'module.base.layer3.4.bn2', 'module.base.layer3.4.bn3', 
                    'module.base.layer3.5.bn1', 'module.base.layer3.5.bn2', 'module.base.layer3.5.bn3', 'module.base.layer4.0.bn1', 
                    'module.base.layer4.0.bn2', 'module.base.layer4.0.bn3', 'module.base.layer4.0.downsample.1', 'module.base.layer4.1.bn1', 
                    'module.base.layer4.1.bn2', 'module.base.layer4.1.bn3', 'module.base.layer4.2.bn1', 'module.base.layer4.2.bn2', 
                    'module.base.layer4.2.bn3', 'module.feat_bn']
    abn_modules = ['module.base.layer4.0.bn1', 
                    'module.base.layer4.0.bn2', 'module.base.layer4.0.bn3', 'module.base.layer4.0.downsample.1', 'module.base.layer4.1.bn1', 
                    'module.base.layer4.1.bn2', 'module.base.layer4.1.bn3', 'module.base.layer4.2.bn1', 'module.base.layer4.2.bn2', 
                    'module.base.layer4.2.bn3', 'module.feat_bn']

    abn_param = ABN_Parameters()
    for n, m in model.named_modules():
        if m.__class__.__name__.find('BatchNorm') != -1 and n in abn_modules:

            abn_param.reset(m.num_features)
            end = time.time()

            for i, (imgs, fnames, pids, _) in enumerate(data_loader):
                data_time.update(time.time() - end)

                bn_res = extract_bn_responses(model, imgs, m)
                c = bn_res.size(1)
                bn_res = bn_res.transpose_(1,-1).contiguous().view(-1,c)
                mean = bn_res.mean(0)
                var = bn_res.var(0)
                abn_param.update(mean, var, data_loader.batch_size)

                batch_time.update(time.time() - end)
                end = time.time()

                if (i + 1) % print_freq == 0:
                    print('Extract {} Respones: [{}/{}]\t'
                          'Time {:.3f} ({:.3f})\t'
                          'Data {:.3f} ({:.3f})\t'
                          .format(n, i + 1, len(data_loader),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg))

            model.state_dict()[n+'.running_mean'].copy_(abn_param.running_mean)
            model.state_dict()[n+'.running_var'].copy_(abn_param.running_var)

class Evaluator_ABN(object):
    def __init__(self, model, dataset='market1501'):
        super(Evaluator_ABN, self).__init__()
        self.model = model
        self.dataset = dataset

    def evaluate(self, data_loader, query, gallery, metric=None):
        adapt_source_bn(self.model, data_loader)

        features, _ = extract_features(self.model, data_loader)
        distmat = pairwise_distance(features, query, gallery, metric=metric)
        return evaluate_all(distmat, query=query, gallery=gallery, dataset=self.dataset)
