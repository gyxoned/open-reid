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

def extract_cnn_feature_adapt(model, s_inputs, t_inputs, modules=None):
    model.eval()
    s_inputs = to_torch(s_inputs)
    t_inputs = to_torch(t_inputs)
    s_inputs = Variable(s_inputs, volatile=True)
    t_inputs = Variable(t_inputs, volatile=True)
    if modules is None:
        outputs, _ = model(s_inputs, t_inputs)
        outputs = outputs.data.cpu()
        return outputs

def extract_bn_responses(model, input, modules):
    model.eval()
    input = to_torch(input)
    input = Variable(input, volatile=True).cuda()

    inputs = []
    def func(m, i): inputs.append(i[0].data)
    handle=modules.register_forward_pre_hook(func)
    model.module.forward(input)
    handle.remove()
    return inputs[0]
