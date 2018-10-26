from __future__ import absolute_import
from collections import OrderedDict

from torch.autograd import Variable

from ..utils import to_torch

def extract_cnn_feature(model, inputs, modules=None):
    model.eval()
    inputs = to_torch(inputs)
    inputs = Variable(inputs, volatile=True)
    if modules is None:
        outputs = model(inputs)
        outputs = outputs.data.cpu()
        return outputs
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())

def extract_bn_responses(model, input):
    model.eval()
    input = to_torch(input)
    input = Variable(input, volatile=True).cuda()

    outputs = OrderedDict()
    inputs = OrderedDict()
    module_name = OrderedDict()
    handles = []
    for n, m in model.named_modules():
        module_name[id(m)] = n
        if m.__class__.__name__.find('BatchNorm') != -1:
            inputs[id(m)] = None
            outputs[id(m)] = None
            def func(m, i, o): 
                inputs[id(m)] = i[0].data.cpu()
                outputs[id(m)] = o.data.cpu()
            handles.append(m.register_forward_hook(func))
    model.module.forward(input)
    for h in handles:
        h.remove()
    return list(inputs.values()), list(outputs.values())