# -*- coding: utf-8 -*-
# @Author: Jingyi
# @Date:   2020-07-20 23:45:20
# @Last Modified by:   jingyi
# @Last Modified time: 2020-12-03 22:36:40


import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    
    def __init__(self, code_size):
        super(AutoEncoder, self).__init__()

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
            self.classifier = nn.Linear(16*14, code_size)
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
            self.classifier = nn.Linear(16*28, code_size)

        
    def forward(self, features):
        images = features.view((features.shape[0], 1, features.shape[-1]))
        # import pdb; pdb.set_trace()
        code = self.encoder(images)
        out = self.decoder(code)
        out = out.view((out.shape[0], -1))
        out = self.classifier(out)
        return out
