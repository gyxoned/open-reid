from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

class SupervisedClusteringLoss(nn.Module):
    """
        supervise cluster loss, proposed by Marc et. al, ICML2017.
        inputs are the feature matrix and ground truth matrix.
        return a scalar loss.
        Attributes:
    """
    def __init__(self):
        super(SupervisedClusteringLoss, self).__init__()
        
    def forward(self, F, Y):
        ###solve C = Y * pinv_Y
        Y = Y.type(torch.FloatTensor).cuda()
        diagy = torch.sum(Y, 0)
        mask = diagy.lt(1)
        diagy = torch.div(torch.ones(diagy.size()).cuda(), diagy.masked_fill_(mask, 1))
        diagy = torch.diag(diagy)
        J = torch.mm(Y, diagy)
        C = torch.mm(Y, J.transpose(0,1))
        torch.norm(torch.mm(torch.mm(Y, J.transpose(0,1)), Y) - Y)
        ###solve pinv_F
        U,S,V = torch.svd(F.data)
        Slength = float(S.size(0))
        maxS = 1e-15 * torch.max(S) * Slength
        maskS = S.le(maxS)
        ST = torch.div(torch.ones(S.size()).cuda(), S)
        ST = ST.masked_fill_(maskS, 0)
        pinvF = torch.mm(U, torch.diag(ST))
        pinvF = torch.mm(pinvF, V.transpose(0,1))
        pinvF = pinvF.transpose(0,1)
        ##solve the loss
        FJ = torch.mm(pinvF, C)
        G = torch.mm(torch.mm(F.data, FJ) - C, FJ.transpose(0,1))
        G = Variable(G, requires_grad=False)
        loss = torch.sum(G * F)
        return loss
        
        
        
        
        
        
        
