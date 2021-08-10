# -*- coding: utf-8 -*-
# @Author: Jingyi
# @Date:   2020-07-08 00:17:40
# @Last Modified by:   jingyi
# @Last Modified time: 2021-03-11 22:30:52

import torch
from torch import optim
import torch.nn as nn
import pandas as pd
import pdb
from torch.utils.data import DataLoader
from dataset import unsw, cicids

from ae import AE
from vae import VAE
from cnnae import AutoEncoder
# from ensemble import Ensemble
from mae import MAE
from mahalanobis import MahalanobisLayer
from cnnvae import VAutoEncoder


from tqdm import tqdm
from fastloader import FastTensorDataLoader
from torch.optim import lr_scheduler
from sklearn.metrics.pairwise import paired_distances
from sklearn.metrics import mean_squared_error
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support

# torch.cuda.set_device('1')


SEED =0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

CUDA_LAUNCH_BLOCKING=0
batch_size=1024

# train_x, train_y = unsw("/home/jingyi/Avada/unsw/UNSW-NB15_rs42_training.csv", is_train=True)
train_x, train_y = cicids("/home/jingyi/Avada/cicids/cicids2017_rs1_training.csv", is_train=True)

# test_x, test_y = unsw("/home/jingyi/Avada/unsw/UNSW-NB15_rs42_test.csv", is_train=False)
test_x, test_y = cicids("/home/jingyi/Avada/cicids/cicids2017_rs1_test.csv", is_train=False)


train_loader= FastTensorDataLoader(train_x, train_y, batch_size=batch_size, shuffle=True)
# val_loader= FastTensorDataLoader(val_x, val_y, batch_size=batch_size, shuffle=True)
test_loader = FastTensorDataLoader(test_x, test_y, batch_size=batch_size, shuffle=True)
# val_size = val_x.size(0)
test_size = test_x.size(0)

# model = AE(input_shape=43).cuda()
# model = VAE(feature_dim=43).cuda()
# model = Ensemble().cuda()
# model = AutoEncoder(43).cuda()

# layer_dims = 43, 512, 256, 128, 64, 32, 16, 32, 64, 128, 256, 512, 43
feature_dim = 82 #41

model = AutoEncoder(feature_dim).cuda()
mahalanobis_cov_decay=0.001
mahalanobis_layer = MahalanobisLayer(feature_dim, mahalanobis_cov_decay).cuda()

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)


# mean-squared error loss
criterion = nn.MSELoss()
best_p=0.2

#BCELoss (Binary Cross Entropy)
# criterion = nn.BCELoss()
# import pdb; pdb.set_trace()

# print("cicids-cnnvae")
def save_checkpoint(state, filename='cicids_cnnae_mse.pth.tar'):
    torch.save(state, filename)

def train(epoch, train_loader):
    model.train()
    loss = 0.0
    for idx, (batch_features, _) in enumerate(tqdm(train_loader)):

        batch_features = batch_features.view(-1, feature_dim)
        # import pdb; pdb.set_trace()
        # input_features = Variable(batch_features.cuda(), requires_grad=True)
        input_features = batch_features.cuda()
        # import pdb; pdb.set_trace()
        optimizer.zero_grad()
        output = model(input_features)
        # output = output.view(-1, feature_dim)

        # print(model.decoder_hidden_layer.weight)
        # if mahalanobis:

        # maha_distance = mahalanobis_layer(input_features, output)
        # import pdb; pdb.set_trace()
        # maha_distance = maha_distance.view(-1, 43)
        # import pdb; pdb.set_trace()
        # 
        train_loss = criterion(output, input_features)
        # train_loss = torch.mean(maha_distance)
        train_loss.backward()
        # import pdb; pdb.set_trace()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        loss += train_loss.item()

    loss = loss / len(train_loader)
    print("loss = {:.6f}".format(loss))
    return loss



def testing(test_loader):
    model.eval()
    i= 0
    result = np.zeros([test_size, 1])
    global best_p
    for idx, (feature, label) in enumerate(tqdm(test_loader)):
        input_feature = feature[:, :feature_dim]
        category = feature[:, feature_dim]
        input_feature = input_feature.cuda()
        # input_feature = feature.view(-1, 1, 41)
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            output = model(input_feature)
        output = output.view(-1, feature_dim)
        if i==0:
            outputs = output
            features = input_feature
            labels = label
            categorys = category
        else:
            outputs = torch.cat((outputs, output), 0)
            features = torch.cat((features, input_feature), 0)
            labels = torch.cat((labels, label), 0)
            categorys = torch.cat((categorys, category), 0)
        i+=1
    outputs = outputs.cpu().detach().numpy()
    features = features.cpu().detach().numpy()
    labels = labels.numpy()
    categorys = categorys.numpy()
    distances = paired_distances(features, outputs)

    result = np.where(distances < np.percentile(distances, 89), 0, 1)
    # import pdb; pdb.set_trace()
    df_from_arr = pd.DataFrame(data=[categorys, result, labels]).T
    cat_error = ((df_from_arr.groupby([0]).sum()[1]-df_from_arr.groupby([0]).sum()[2]).abs()/df_from_arr.groupby([0]).count()[1]).to_numpy()

    acc = np.sum(result==labels)/test_size
    # import pdb; pdb.set_trace()
    p, r, f, _ = precision_recall_fscore_support(labels, result, average='macro')

    x=p-best_p
    if x>0:
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
            }
        )
        print("saving checkpoint")
        print("cat_error", cat_error)
        best_p = p
    return acc, p, r,f


for epoch in range(50):
    print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
    loss=train(epoch, train_loader=train_loader)
    # print(best_loss)
    scheduler.step(loss)
    # epoch += 1
    if (epoch+1)%5==0:
        acc, p, r, f = testing(test_loader)
        print('acc:{:.6f}, precision:{:.6f}, recall:{:.6f}, f1:{:.6f}'.format(acc, p, r, f))

'''
acc, p, r, f = testing(test_loader)
print('acc:{:.6f}, precision:{:.6f}, recall:{:.6f}, f1:{:.6f}'.format(acc, p, r, f))
'''