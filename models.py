import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import h5py as h5
from scipy import stats

import torch
from torch import FloatTensor
from torch.autograd import Variable
from torch.nn import Linear, Module, BatchNorm1d, Dropout
from torch.nn.functional import sigmoid, softmax

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
            # e_zs = stats.zscore(e, axis=0)
            query = FloatTensor(np.array(f[image_model][layer][idx * 8: idx * 8 + 8]))
            target = FloatTensor(e[:, 66])
            minibatches.append((query, target))
        return minibatches



class Fetures2ECoGTrans(Module):
    def __init__(self, features_dim, hidden_dim):
        super(Fetures2ECoGTrans, self).__init__()
        self.hidden_dim = hidden_dim
        self.features_dim = features_dim
        self.f1 = Linear(self.features_dim, 1)
        self.d = Dropout(p=0.3)
        # self.f2 = Linear(self.hidden_dim, 1)
        # self.b = BatchNorm1d(self.hidden_dim)

    def forward(self, tensor, **kwargs):

        tensor_flat = tensor.view(tensor.shape[0], tensor.shape[1] * tensor.shape[2], tensor.shape[3])
        w, tensor_att = attend(1, tensor_flat, return_weight=True, value=tensor_flat, **kwargs)
        tensor_att = self.d(tensor_att)
        norm = tensor_att.norm(p=2, dim=1, keepdim=True)
        tensor_lin = self.f1(tensor_att)
        # tensor_lin = self.b(tensor_lin)
        # tensor_lin = self.f2(tensor_lin)
        # tensor_lin_flat = tensor_lin.view((tensor_lin.shape[0] * tensor_lin.shape[1]))
        # tensor_bn = self.b(tensor_lin)
        # tensor_out = tensor_sm.view((tensor_lin.shape[0], tensor_lin.shape[1]))
        return w, torch.div(tensor_lin, norm)

