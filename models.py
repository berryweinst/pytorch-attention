import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import h5py as h5
from scipy import stats

import torch
from torch import FloatTensor
from torch.autograd import Variable
from torch.nn import Linear, Module, BatchNorm1d

from attention import attend

class Data(object):

    @staticmethod
    def create_minibatches(image_model, layer, batch_size):
        ecog_data = pickle.load(open("data_dict_src.p", "rb"))
        ecog_data = ecog_data[0:int(ecog_data.shape[0] / batch_size) * batch_size, :].reshape(
            (int(ecog_data.shape[0] / batch_size), batch_size, ecog_data.shape[1]))
        f = h5.File("features.h5", "r")
        minibatches = []
        for idx, e in enumerate(ecog_data):
            target = stats.zscore(np.array(f[image_model][layer][idx * 8: idx * 8 + 8]), axis=3)
            query = FloatTensor(e)
            target = FloatTensor(target)
            minibatches.append((query, target))
        return minibatches



class Fetures2ECoGTrans(Module):
    def __init__(self, features_dim, hidden_dim):
        super(Fetures2ECoGTrans, self).__init__()
        self.hidden_dim = hidden_dim
        self.features_dim = features_dim
        self.f = Linear(self.features_dim, self.hidden_dim)
        self.b = BatchNorm1d(self.hidden_dim)

    def forward(self, tensor, **kwargs):

        tensor_flat = tensor.view((tensor.shape[0], tensor.shape[1] * tensor.shape[2], tensor.shape[3]))
        tensor_att = attend(1, tensor_flat, value=tensor_flat, **kwargs)
        tensor_lin = self.f(tensor_att)
        # tensor_lin_flat = tensor_lin.view((tensor_lin.shape[0] * tensor_lin.shape[1]))
        tensor_bn = self.b(tensor_lin)
        # tensor_out = tensor_sm.view((tensor_lin.shape[0], tensor_lin.shape[1]))
        return tensor_bn

