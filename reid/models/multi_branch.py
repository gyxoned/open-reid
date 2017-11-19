from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pdb


def random_walk_compute(p_g_score, g_g_score, alpha):
    # Random Walk Computation
    one_diag = Variable(torch.eye(g_g_score.size(0)).cuda(), requires_grad=False)
    g_g_score_sm = Variable(g_g_score.data.clone(), requires_grad=False)
    # Row Normalization
    A = F.softmax(g_g_score_sm[:, :, 1].squeeze())
    A = (1 - alpha) * torch.inverse(one_diag - alpha * A)
    A = A.transpose(0, 1)
    p_g_score = torch.matmul(p_g_score.permute(2, 0, 1), A).permute(1, 2, 0).contiguous()
    g_g_score = torch.matmul(g_g_score.permute(2, 0, 1), A).permute(1, 2, 0).contiguous()
    p_g_score = p_g_score.view(-1, 2)
    g_g_score = g_g_score.view(-1, 2)
    outputs = torch.cat((p_g_score, g_g_score), 0)
    outputs = outputs.contiguous()

    return outputs


class RandomWalkNet(nn.Module):
    def __init__(self, instances_num=4, base_model=None, embed_model=None):
        super(RandomWalkNet, self).__init__()
        self.instances_num = instances_num
        self.base = base_model
        self.embed = embed_model
        self.topk = 10

    def forward(self, x):
        x = self.base(x)
        N, C = x.size()
        x = x.view(int(N / self.instances_num), self.instances_num, -1)

        probe_x = x[:, 0, :]
        probe_x = probe_x.contiguous()
        probe_x = probe_x.view(-1, C)
        gallery_x = x[:, 1:self.instances_num, :]
        gallery_x = gallery_x.contiguous()
        gallery_x = gallery_x.view(-1, C)

        p_g_score = self.embed(probe_x, gallery_x)
        g_g_score = self.embed(gallery_x, gallery_x)
        
        #Random Walk Computation
        alpha = 0.5
        ones = Variable(torch.ones(g_g_score.size()[:2]), requires_grad=False).cuda()
        one_diag = Variable(torch.eye(g_g_score.size(0)), requires_grad=False).cuda()
        D = torch.diag(1.0 / torch.sum((ones - one_diag) * g_g_score[:,:,1], 1))
        A = torch.matmul(D, g_g_score[:,:,1])
        A = (1 - alpha) * torch.inverse(one_diag - alpha * A)
        A = A.transpose(0,1)
        p_g_score[:,:,1] = torch.matmul(p_g_score[:,:,1].clone(), A)
        p_g_score = p_g_score.view(-1, 2)
        g_g_score = g_g_score.view(-1, 2)
        outputs = torch.cat((p_g_score, g_g_score), 0)
        outputs = outputs.contiguous()

        return outputs


class RandomWalkNetGrp(nn.Module):
    def __init__(self, instances_num=4, base_model=None, embed_model=None, alpha=0.1):
        super(RandomWalkNetGrp, self).__init__()
        self.instances_num = instances_num
        self.alpha = alpha
        self.base = base_model
        self.embed = embed_model
        for i in range(len(embed_model)):
            setattr(self, 'embed_'+str(i), embed_model[i])

    def forward(self, x):
        x = self.base(x)
        N, C = x.size()
        x = x.view(int(N / self.instances_num), self.instances_num, -1)

        probe_x = x[:, 0, :]
        probe_x = probe_x.contiguous()
        probe_x = probe_x.view(-1, C)
        gallery_x = x[:, 1:self.instances_num, :]
        gallery_x = gallery_x.contiguous()
        gallery_x = gallery_x.view(-1, C)
        count = 2048 / (len(self.embed))
        outputs = []
        for i in range(len(self.embed)):
            p_g_score = self.embed[i](probe_x[:,i*count:(i+1)*count].contiguous(),
                                      gallery_x[:,i*count:(i+1)*count].contiguous())
            g_g_score = self.embed[i](gallery_x[:,i*count:(i+1)*count].contiguous(),
                                      gallery_x[:,i*count:(i+1)*count].contiguous())

            outputs.append(random_walk_compute(p_g_score, g_g_score, self.alpha))

        outputs = torch.cat(outputs, 0)
        return outputs


class SiameseNet(nn.Module):
    def __init__(self, base_model, embed_model):
        super(SiameseNet, self).__init__()
        self.base_model = base_model
        self.embed_model = embed_model

    def forward(self, x1, x2):
        x1, x2 = self.base_model(x1), self.base_model(x2)
        if self.embed_model is None:
            return x1, x2
        return self.embed_model(x1, x2)
