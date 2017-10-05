from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pdb


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
