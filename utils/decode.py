import torch


def multi_decode(data):
    pre = []
    target = []
    for item in data:
        outputs, labels = item
        outputs = torch.argmax(outputs, dim=1).detach().tolist()
        pre.extend(outputs)
        target.extend(labels.tolist())
    return pre, target
