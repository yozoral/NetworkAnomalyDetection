# -*- coding: utf-8 -*-
# @Author: Jingyi
# @Date:   2020-09-03 21:29:31
# @Last Modified by:   jingyi
# @Last Modified time: 2020-11-19 22:20:06

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class VAutoEncoder(nn.Module):
    
    def __init__(self, code_size):
        super(VAutoEncoder, self).__init__()

        if code_size==41:
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels = 1, out_channels = 32, kernel_size = 5, stride=5, padding=3),
                nn.ReLU(),
                nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
                )

            self.decoder = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0),
                nn.Upsample(scale_factor=2, mode='linear'),
                nn.ReLU()
                )
            self.fc1 = nn.Linear(9, 8)
            self.fc2 = nn.Linear(9, 8)
            self.classifier = nn.Linear(16*12, 41)

        elif code_size==82:
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels = 1, out_channels = 32, kernel_size = 5, stride=5, padding=0),
                nn.ReLU(),
                nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
                )

            self.decoder = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0),
                nn.Upsample(scale_factor=2, mode='linear'),
                nn.ReLU()
                )
            self.fc1 = nn.Linear(16, 8)
            self.fc2 = nn.Linear(16, 8)
            self.classifier = nn.Linear(16*12, code_size)
        
    def forward(self, x):
        x = x.view((x.shape[0], 1, x.shape[-1]))
        # import pdb; pdb.set_trace()
        x = self.encoder(x)
        logvar = self.fc1(x)
        mu = self.fc2(x)
        z = self.reparametrize(mu, logvar)
        z = self.decoder(z)
        z = z.view((z.shape[0], -1))
        z = self.classifier(z)
        return z
    
    # def encode(self, features):
    #     x = F.relu(self.conv1(features))
    #     x = F.relu(self.conv2(x))
    #     return self.conv3(x), self.conv4(x)
    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    # def decode(self, z):
    #     # import pdb; pdb.set_trace()
    #     out = F.relu(self.conv_trans1(z))
    #     out = F.relu(self.conv_trans2(out))
    #     out = F.sigmoid(self.conv_trans3(out))
    #     return out