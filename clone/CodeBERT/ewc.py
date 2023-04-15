import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss
import copy
#from model.masked_cross_entropy import *
def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

class EWC(object):
    def __init__(self, model, dataset, device, length):

        self.model = model
        # the data we use to compute fisher information of ewc (old_exemplars)
        self.dataset = dataset
        self.device = device
        self.length = length

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {} # previous parameters
        self._precision_matrices = self._diag_fisher() # approximated diagnal fisher information matrix

        for n, p in copy.deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):

        self.model.train()
        precision_matrices = {}
        for n, p in copy.deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        for step,batch in enumerate(self.dataset):
            self.model.zero_grad()
            
            inputs = batch[0].to(self.device)        
            labels=batch[1].to(self.device) 
            self.model.train()
            loss,logits = self.model(inputs,labels)

            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                precision_matrices[n].data += p.grad.data ** 2 / self.length

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        self.model.zero_grad()
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()

        return loss
