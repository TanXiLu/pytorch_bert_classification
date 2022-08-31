# coding=utf-8
import torch.nn as nn


def cal_loss(predict, data):
    loss_function = nn.CrossEntropyLoss()
    loss = loss_function(predict, data["labels"])
    return loss
