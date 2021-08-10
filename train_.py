# -*- coding: utf-8 -*-
# @Author: Jingyi
# @Date:   2020-07-08 00:17:40
# @Last Modified by:   jingyi
# @Last Modified time: 2020-09-29 00:30:06

import itertools
import torch

from torch import optim
import torch.nn as nn
import pandas as pd
import pdb
from torch.utils.data import DataLoader
from dataset import unsw

from gan_network import ALAD, encoder, decoder, discriminator_xx, discriminator_xz, discriminator_zz

from tqdm import tqdm
from fastloader import FastTensorDataLoader
from sklearn.metrics.pairwise import paired_distances
# from sklearn.metrics import mean_squared_error
import numpy as np
from torch.autograd import Variable
from lossF import encLoss, discriLoss

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

torch.autograd.set_detect_anomaly(True)
# torch.cuda.set_device('0')

CUDA_LAUNCH_BLOCKING=0
batch_size=100

train_x, train_y = unsw("./unsw/train.csv")
# val_x, val_y = unsw("./unsw/val.csv")
test_x, test_y = unsw("./unsw/test.csv")
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

input_dim=43
latent_dim=8

model = ALAD(input_dim=43, latent_dim=8).cuda()


inf = encoder(input_dim, latent_dim).cuda()
gen = decoder(latent_dim, input_dim).cuda()
dn_xx = discriminator_xx(input_dim, input_dim).cuda()
dn_zz = discriminator_zz(latent_dim, latent_dim).cuda()
dn_xz = discriminator_xz(input_dim, latent_dim).cuda()

# mahalanobis_cov_decay=0.001

# create an optimizer object
# Adam optimizer with learning rate 1e-3

# optimizer_enc = optim.Adam(model.enc.parameters(), lr=1e-3)
# optimizer_gen = optim.Adam(model.gen.parameters(), lr=1e-3)
optimizer_enc_gen = optim.Adam((list(model.gen.parameters()) + list(model.enc.parameters())), lr = 1e-3, betas = (0.5,0.999))
optimizer_D = optim.Adam(itertools.chain(model.dis_xx.parameters(), model.dis_xz.parameters(), model.dis_zz.parameters()), lr = 1e-4, betas = (0.5,0.999))
# optimizer_xz = optim.Adam(model.dis_xz.parameters(), lr=1e-3)
# optimizer_xx = optim.Adam(model.dis_xx.parameters(), lr=1e-3)
# optimizer_zz = optim.Adam(model.dis_zz.parameters(), lr=1e-3)

# optimizer = optim.Adam(model.parameters(), lr=1e-4)
#BCELoss (Binary Cross Entropy)
# criterion = nn.BCELoss()


# criterion = nn.BCEWithLogitsLoss()
lossF = encLoss()
lossF_dis = discriLoss()


def train(epoch, train_loader, allow_zz=True):
    '''
    enc.train()
    gen.train()
    dis_zz.train()
    dis_xz.train()
    dis_xx.train()
    '''
    # model.train()
    bce = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    loss_dis, loss_gen = 0.0, 0.0
    for idx, (batch_features, _) in enumerate(tqdm(train_loader)):

        batch_features = batch_features.view(-1, 43)
        # import pdb; pdb.set_trace()
        # input_features = batch_features
        # import pdb; pdb.set_trace()

        latent_features = torch.from_numpy(np.random.normal(size=[batch_size, 8])).float().cuda()

        
        #########################################################################

        if (epoch+1)%1==0:

            # optimizer_gen.zero_grad()
            # optimizer_enc.zero_grad()
            # optimizer.zero_grad()

            x = Variable(batch_features.cuda(), requires_grad=True)
            z = Variable(latent_features.cuda(), requires_grad=True)

            # x_logit_real, x_logit_fake, z_logit_real, z_logit_fake, l_generator, l_encoder = model(x_input.clone(), z_input.clone())
# 
            # loss_gen_enc, loss_gen, loss_enc = lossF(x_logit_real, x_logit_fake, z_logit_real, z_logit_fake, l_generator, l_encoder)
            # dis_loss, dis_loss_xz,dis_loss_xx,dis_loss_zz = lossF_dis(x_logit_real, x_logit_fake, z_logit_real, z_logit_fake, l_generator, l_encoder)

            p_x = gen(z)
            q_z = inf(x)
        
          
            decoder_logit, _ = dn_xz(p_x, z)
            encoder_logit, _ = dn_xz(x , q_z)
    
          
            decoder_loss = bce(sigmoid(decoder_logit),torch.zeros_like(decoder_logit))
            encoder_loss =  bce(sigmoid(encoder_logit),torch.ones_like(encoder_logit))
            dis_loss_xz = torch.mean(encoder_loss+decoder_loss)
            decoder_loss2 =  bce(sigmoid(decoder_logit),torch.ones_like(decoder_logit))
            encoder_loss2 = bce(sigmoid(encoder_logit),torch.zeros_like(encoder_logit))
          
            gen_loss_xz = torch.mean((decoder_loss2))  + (torch.mean(encoder_loss2))
          
            rec_z = inf(p_x)
            rec_x = gen(q_z)
           
            x_logit_real,_ = dn_xx(x,x)
            x_logit_fake,_ = dn_xx(x,rec_x)
            z_logit_real,_ = dn_zz(z,z)
            z_logit_fake,_ = dn_zz(z,rec_z)
        
            x_sigmoid_real = bce(sigmoid(x_logit_real),torch.ones_like(x_logit_real))
            x_sigmoid_fake = bce(sigmoid(x_logit_fake),torch.zeros_like(x_logit_fake))
            x_sigmoid_real2 = bce(sigmoid(x_logit_real),torch.zeros_like(x_logit_real))
            x_sigmoid_fake2 = bce(sigmoid(x_logit_fake),torch.ones_like(x_logit_fake))
            z_sigmoid_real = bce(sigmoid(z_logit_real),torch.ones_like(z_logit_real))
            z_sigmoid_fake = bce(sigmoid(z_logit_fake),torch.zeros_like(z_logit_fake))
            z_sigmoid_real2 = bce(sigmoid(z_logit_real),torch.zeros_like(z_logit_real))
            z_sigmoid_fake2 = bce(sigmoid(z_logit_fake),torch.ones_like(z_logit_fake))
          
            dis_loss_x = torch.mean(x_sigmoid_real + x_sigmoid_fake)
            dis_loss_z = torch.mean(z_sigmoid_real + z_sigmoid_fake)
            dis_loss = dis_loss_xz + dis_loss_x + dis_loss_z
          
            cost_x = torch.mean(x_sigmoid_real2 + x_sigmoid_fake2) 
            cost_z = torch.mean(z_sigmoid_real2 + z_sigmoid_fake2)
            loss_gen_enc = gen_loss_xz + cost_x  + cost_z
    


            optimizer_enc_gen.zero_grad()

            loss_gen_enc.backward(retain_graph = True)
            # print(model.gen.layer1.weight)
            # import pdb; pdb.set_trace()
            
            # optimizer_enc.step()
            # optimizer_gen.step()
            optimizer_enc_gen.step()
            # loss_encoder_s += loss_enc.item()
            # loss_decoder_s += loss_gen.item()


            optimizer_D.zero_grad()
            # optimizer_xz.step()
            # optimizer_xx.step()
            # optimizer_zz.step()

            dis_loss.backward()

            optimizer_D.step()

            loss_gen += loss_gen_enc.item()
            loss_dis += dis_loss.item()
            # loss_zz_s += dis_loss_zz.item()


        # loss += train_loss.item()

    loss_gen = loss_gen / len(train_loader)
    # loss_decoder_s = loss_decoder_s / len(train_loader)
    # loss_xx_s = loss_xx_s / len(train_loader)
    # loss_xz_s = loss_xz_s / len(train_loader)
    loss_dis = loss_dis / len(train_loader)

    # print("epoch : {}, loss_enc = {:.6f}, loss_dec = {:.6f}, loss_xx = {:.6f}, loss_xz = {:.6f}, loss_zz = {:.6f}".format(epoch + 1, loss_encoder_s, loss_decoder_s, loss_xx_s, loss_xz_s, loss_zz_s))
    print("epoch : {}, loss_enc_gen = {:.6f}, loss_xz = {:.6f}".format(epoch + 1, loss_gen, loss_dis))


'''
def validation(epoch, val_loader):
    model.eval()
    val_loss, i=0.0, 0
    result = np.zeros([val_size, 1])
    for idx, (feature, label) in enumerate(tqdm(val_loader)):
        feature = feature.cuda()
        output = model(feature)
        val_loss += batch_size*criterion(output, feature).item()
        # correct += label.eq(output.max(1)[1]).cpu().sum()
        if i==0:
            outputs = output
            features = feature
            labels = label
        else:
            outputs = torch.cat((outputs, output), 0)
            features = torch.cat((features, feature), 0)
            labels = torch.cat((labels, label), 0)
        i+=1
    outputs = outputs.cpu().detach().numpy()
    features = features.cpu().detach().numpy()
    labels = labels.numpy()
    distances = paired_distances(features, outputs)

    result = np.where(distances < np.percentile(distances, 90), 0, 1)

    acc = np.sum(result==labels)/val_size
    model.train()
    val_loss /= val_size
    print("Valid Epoch: {} \tLoss: {:.6f}\tAcc:{:.6f}".format(epoch, val_loss, acc))
'''


'''
def testing(test_loader):
    model.eval()
    i= 0
    result = np.zeros([test_size, 1])
    for idx, (feature, label) in enumerate(tqdm(test_loader)):
        feature = feature.cuda()
        input_feature = feature.view(-1, 43)
        with torch.no_grad():
            output = model(input_feature)
        output = output.view(-1, 43)
        if i==0:
            outputs = output
            features = feature
            labels = label
        else:
            outputs = torch.cat((outputs, output), 0)
            features = torch.cat((features, feature), 0)
            labels = torch.cat((labels, label), 0)
        i+=1
    outputs = outputs.cpu().detach().numpy()
    features = features.cpu().detach().numpy()
    labels = labels.numpy()
    distances = paired_distances(features, outputs)

    result = np.where(distances < np.percentile(distances, 90), 0, 1)

    acc = np.sum(result==labels)/test_size
    return acc
'''

def testing(test_loader):
    i= 0
    result = np.zeros([test_size, 1])
    for idx, (feature, label) in enumerate(tqdm(test_loader)):
        feature = feature.cuda()
        input_feature = feature.view(-1, 43)
        #import pdb; pdb.set_trace()
        latent_features = torch.from_numpy(np.random.normal(size=[batch_size, 8])).float()
        with torch.no_grad():
            output = model(input_feature, latent_features, is_training=False)
            # z_gen = enc(input_feature)
            # output = gen(z_gen)
        # output = output.view(-1, 43)
        if i==0:
            outputs = output
            features = feature
            labels = label
        else:
            outputs = torch.cat((outputs, output), 0)
            features = torch.cat((features, feature), 0)
            labels = torch.cat((labels, label), 0)
        i+=1
    outputs = outputs.cpu().detach().numpy()
    features = features.cpu().detach().numpy()
    labels = labels.numpy()
    distances = paired_distances(features, outputs)

    result = np.where(distances < np.percentile(distances, 90), 0, 1)

    acc = np.sum(result==labels)/test_size
    return acc

best_acc=0
for epoch in range(50):
    train(epoch, train_loader=train_loader)
    # print(best_loss
    acc = testing(test_loader)
    if acc> best_acc:
        best_acc = acc
    print('acc:', acc)
    epoch += 1

print(best_acc)


