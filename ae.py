# -*- coding: utf-8 -*-
# @Author: Jingyi
# @Date:   2020-07-08 00:16:18
# @Last Modified by:   jingyi
# @Last Modified time: 2020-10-12 23:06:32

import torch
import torch.nn as nn
from mahalanobis import MahalanobisLayer

class AE(nn.Module):
    def __init__(self, input_shape, mahalanobis=False, mahalanobis_cov_decay=0.1):
        super(AE, self).__init__()
        # self.encoder_hidden_layer = nn.Linear(
        #     in_features=input_shape, out_features=128
        # )
        # self.encoder_output_layer = nn.Linear(
        #     in_features=128, out_features=128
        # )
        # self.decoder_hidden_layer = nn.Linear(
        #     in_features=128, out_features=128
        # )
        # self.decoder_output_layer = nn.Linear(
        #     in_features=128, out_features=input_shape
        # )

        self.encoder_layer = nn.Sequential(
            nn.Linear(input_shape, 1536),
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
            nn.Linear(16, 8)
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
            nn.Linear(1536, input_shape)
            )

        self.mahalanobis = mahalanobis

        if mahalanobis:
            self.mahalanobis_layer = MahalanobisLayer(input_shape,
                                                      mahalanobis_cov_decay)

    def forward(self, features):
        # activation = self.encoder_hidden_layer(features)
        # activation = torch.relu(activation)
        # code = self.encoder_output_layer(activation)
        # code = torch.relu(code)
        # activation = self.decoder_hidden_layer(code)
        # activation = torch.relu(activation)
        # activation = self.decoder_output_layer(activation)
        # reconstructed = torch.relu(activation)
        latent = self.encoder_layer(features)
        reconstructed = self.decoder_layer(latent)

        if self.mahalanobis:
            reconstructed = self.mahalanobis_layer(features, reconstructed)

        return reconstructed
