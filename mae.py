# -*- coding: utf-8 -*-
# @Author: Institute for Infocomm Research
# @Date:   2020-08-14


import torch
import torch.nn as nn
from mahalanobis import MahalanobisLayer
from dataset import unsw
from fastloader import FastTensorDataLoader
from torch import optim
from tqdm import tqdm
import numpy as np


class MAE(nn.Module):
    def __init__(self, layer_dims):
        super(MAE, self).__init__()

        self.layer_dims = layer_dims

        self.encoding_layers = torch.nn.Sequential(
            nn.Linear(layer_dims[0], layer_dims[1]),  # 1st hidden layer
            nn.Tanh(),  # 1st hidden layer
            nn.Linear(layer_dims[1], layer_dims[2]),  # 2nd hidden layer
            nn.Tanh(),  # 2nd hidden layer
            nn.Linear(layer_dims[2], layer_dims[3]),  # 3rd hidden layer
            nn.Tanh(),  # 3rd hidden layer
            nn.Linear(layer_dims[3], layer_dims[4]),  # 4th hidden layer
            nn.Tanh(),  # 4th hidden layer
            nn.Linear(layer_dims[4], layer_dims[5]),  # 5th hidden layer
            nn.Tanh(),  # 5th hidden layer
            nn.Linear(layer_dims[5], layer_dims[6]),   # Compression layer
        )

        self.decoding_layers = torch.nn.Sequential(
            nn.Linear(layer_dims[6], layer_dims[7]),  # 7th hidden layer
            nn.Tanh(),  # 7th hidden layer
            nn.Linear(layer_dims[7], layer_dims[8]),  # 8th hidden layer
            nn.Tanh(),  # 8th hidden layer
            nn.Linear(layer_dims[8], layer_dims[9]),  # 9th hidden layer
            nn.Tanh(),  # 9th hidden layer
            nn.Linear(layer_dims[9], layer_dims[10]),  # 10th hidden layer
            nn.Tanh(),  # 10th hidden layer
            nn.Linear(layer_dims[10], layer_dims[11]),  # 11th hidden layer
            nn.Tanh(),  # 11th hidden layer
            nn.Linear(layer_dims[11], layer_dims[12])  # Output layer
        )

        # self.mahalanobis = mahalanobis

        # if mahalanobis:
        #     self.mahalanobis_layer = MahalanobisLayer(layer_dims[0],
        #                                               mahalanobis_cov_decay)

    def forward(self, x_feature):

        x_feature_enc = self.encoding_layers(x_feature)
        x_feature_fit = self.decoding_layers(x_feature_enc)

        # if self.mahalanobis:
        #     x_feature_fit = self.mahalanobis_layer(x_feature, x_feature_fit)

        return x_feature_fit

    def reconstruct(self, x_feature):

        x_feature_prime = self.encoding_layers(x_feature)
        x_feature_prime = self.decoding_layers(x_feature_prime)

        return x_feature_prime


'''
if __name__ == "__main__":

    layer_dims = 43, 512, 256, 128, 64, 32, 16, 32, 64, 128, 256, 512, 43

    model = MAE(layer_dims, mahalanobis=True, mahalanobis_cov_decay=0.001).cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = 1024

    train_x, train_y = unsw("unsw/train.csv")
    test_x, test_y = unsw("unsw/test.csv")
    train_loader = FastTensorDataLoader(train_x, train_y, batch_size=batch_size, shuffle=True)
    test_loader = FastTensorDataLoader(test_x, test_y, batch_size=batch_size, shuffle=True)
    test_size = test_x.size(0)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    loss = 0.0
    for idx, (batch_features, _) in enumerate(tqdm(train_loader)):
        batch_features = batch_features.cuda()
        input_features = batch_features.view(-1, 43)
        outputs = model(input_features)
        import pdb; pdb.set_trace()

        train_loss = criterion(outputs, torch.zeros(outputs.size(), device=device))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if model.mahalanobis_layer:
            with torch.no_grad():
                x_fit = model.reconstruct(input_features)
                model.mahalanobis_layer.update(input_features, x_fit)

    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    i = 0
    result = np.zeros([test_size, 1])
    for idx, (feature, label) in enumerate(tqdm(test_loader)):
        test_feature = feature.cuda()
        test_input_feature = test_feature.view(-1, 43)
        with torch.no_grad():
            output = model.reconstruct(test_input_feature)

        if i == 0:
            outputs = output
            features = test_input_feature
            labels = label
        else:
            outputs = torch.cat((outputs, output), 0)
            features = torch.cat((features, test_input_feature), 0)
            labels = torch.cat((labels, label), 0)

        i += 1
'''


