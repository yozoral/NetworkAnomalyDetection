# -*- coding: utf-8 -*-
# @Author: Jingyi
# @Date:   2020-07-17 23:10:09
# @Last Modified by:   jingyi
# @Last Modified time: 2020-11-12 01:52:13

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
# import torchvision
# from torchvision import transforms
# import torch.optim as optim
from torch import nn
from mahalanobis import MahalanobisLayer


# class Normal(object):
#     def __init__(self, mu, sigma, log_sigma, v=None, r=None):
#         self.mu = mu
#         self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
#         self.logsigma = log_sigma
#         dim = mu.get_shape()
#         if v is None:
#             v = torch.FloatTensor(*dim)
#         if r is None:
#             r = torch.FloatTensor(*dim)
#         self.v = v
#         self.r = r


# class Encoder(torch.nn.Module):
#     def __init__(self, D_in, D_out):
#         super(Encoder, self).__init__()
#         self.encoder_layer = nn.Sequential(
#             nn.Linear(D_in, 1536),
#             nn.ReLU(True),
#             nn.Linear(1536, 1024),
#             nn.ReLU(True),
#             nn.Linear(1024, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 384),
#             nn.ReLU(True),
#             nn.Linear(384, 256),
#             nn.ReLU(True),
#             nn.Linear(256, 128),
#             nn.ReLU(True),
#             nn.Linear(128, 64),
#             nn.ReLU(True),
#             nn.Linear(64, 32),
#             nn.ReLU(True),
#             nn.Linear(32, 16),
#             nn.ReLU(True),
#             nn.Linear(16, D_out)
#             )

#     def forward(self, x):
#         return self.encoder_layer(x)


# class Decoder(torch.nn.Module):
#     def __init__(self, D_in, D_out):
#         super(Decoder, self).__init__()
#         self.decoder_layer = nn.Sequential(
#             nn.Linear(D_in, 16),
#             nn.ReLU(True),
#             nn.Linear(16, 32),
#             nn.ReLU(True),
#             nn.Linear(32, 64),
#             nn.ReLU(True),
#             nn.Linear(64, 128),
#             nn.ReLU(True),
#             nn.Linear(128, 256),
#             nn.ReLU(True),
#             nn.Linear(256, 384),
#             nn.ReLU(True),
#             nn.Linear(384, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 1024),
#             nn.ReLU(True),
#             nn.Linear(1024, 1536),
#             nn.ReLU(True),
#             nn.Linear(1536, D_out)
#             )

#     def forward(self, x):
#         return self.decoder_layer(x)


class VAE(torch.nn.Module):

    def __init__(self, feature_dim):
        super(VAE, self).__init__()
        self.encoder_layer = nn.Sequential(
            nn.Linear(feature_dim, 1536),
            nn.ReLU(True),
            nn.Linear(1536, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 384),
            nn.ReLU(True),
            nn.Linear(384, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 100)
            )
        self.decoder_layer = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 384),
            nn.ReLU(True),
            nn.Linear(384, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1536),
            nn.ReLU(True),
            nn.Linear(1536, feature_dim)
            )
        self._enc_mu = torch.nn.Linear(100, 8)
        self._enc_log_sigma = torch.nn.Linear(100, 8)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        # sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=log_sigma.size())).float().cuda()

        self.z_mean = mu
        self.z_sigma = log_sigma
        # import pdb; pdb.set_trace()

        return mu + log_sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder_layer(state)
        z = self._sample_latent(h_enc)
        reconstructed = self.decoder_layer(z)
        return reconstructed
