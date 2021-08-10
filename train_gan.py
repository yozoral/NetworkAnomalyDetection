# -*- coding: utf-8 -*-
# @Author: Jingyi
# @Date:   2020-07-08 00:17:40
# @Last Modified by:   jingyi
# @Last Modified time: 2020-11-18 21:56:52

import itertools
import torch

from torch import optim
import torch.nn as nn
import pandas as pd
import pdb
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from dataset import unsw, cicids

from gan_network import ALAD, encoder, decoder, discriminator_xx, discriminator_xz, discriminator_zz

from tqdm import tqdm
from fastloader import FastTensorDataLoader
from sklearn.metrics.pairwise import paired_distances
# from sklearn.metrics import mean_squared_error
import numpy as np
from torch.autograd import Variable
from lossF import encLoss, discriLoss
from torch.optim import lr_scheduler

SEED =0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

torch.autograd.set_detect_anomaly(True)
# torch.cuda.set_device('0')

# CUDA_LAUNCH_BLOCKING=0
batch_size=100


# train_x, train_y = unsw("/home/jingyi/Avada/unsw/UNSW-NB15_rs42_training.csv", is_train=True)
# # test_x, test_y = unsw("/home/jingyi/Desktop/UNSW-NB15_test.csv")
# test_x, test_y = unsw("/home/jingyi/Avada/unsw/UNSW-NB15_rs42_test.csv", is_train=False)

train_x, train_y = cicids("/home/jingyi/Avada/cicids/cicids2017_rs1_training.csv", is_train=True)
# test_x, test_y = unsw("/home/jingyi/Avada/unsw/UNSW-NB15_rs42_test.csv", is_train=False)
test_x, test_y = cicids("/home/jingyi/Avada/cicids/cicids2017_rs1_test.csv", is_train=False)
print(train_x.shape)
print(test_x.shape)
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

best_p=0.1

input_dim=82
latent_dim=8

model = ALAD(input_dim=input_dim, latent_dim=8).cuda()


optimizer_ = optim.Adam(itertools.chain(model.enc.parameters(), model.gen.parameters()), lr=1e-3)
# optimizer_gen = optim.Adam(model.gen.parameters(), lr=1e-3)
# optimizer_D = optim.Adam(itertools.chain(model.dis_xx.parameters(), model.dis_xz.parameters(), model.dis_zz.parameters()), lr = 1e-4)
optimizer_dis = optim.Adam(itertools.chain(model.dis_xx.parameters(), model.dis_xz.parameters(), model.dis_zz.parameters()), lr=1e-3)
# optimizer_xx = optim.Adam(model.dis_xx.parameters(), lr=1e-3)
# optimizer_zz = optim.Adam(model.dis_zz.parameters(), lr=1e-3)

scheduler1 = lr_scheduler.ReduceLROnPlateau(optimizer_, mode='max', patience=3)
scheduler2 = lr_scheduler.ReduceLROnPlateau(optimizer_dis, mode='max', patience=3)
# scheduler3 = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=6)

lossF = encLoss()
lossF_dis = discriLoss()

def save_checkpoint(state, filename='cicids_gan.pth.tar'):
    torch.save(state, filename)


def train(epoch, train_loader, allow_zz=True):

    model.train()
    loss_encoder_s, loss_decoder_s, loss_xz_s, loss_xx_s, loss_zz_s = 0.0, 0.0, 0.0, 0.0, 0.0
    for idx, (batch_features, _) in enumerate(tqdm(train_loader)):

        batch_features = batch_features.view(-1, input_dim)

        latent_features = torch.from_numpy(np.random.normal(size=[batch_size, 8])).float().cuda()

        
        #########################################################################

        # if (epoch+1)%2!=0:

        # optimizer_gen.zero_grad()
        optimizer_.zero_grad()


        x_input = Variable(batch_features.cuda(), requires_grad=True)
        z_input = Variable(latent_features.cuda(), requires_grad=True)

        x_logit_real, x_logit_fake, z_logit_real, z_logit_fake, l_generator, l_encoder = model(x_input.clone(), z_input.clone())
        loss_gen_enc, loss_gen, loss_enc = lossF(x_logit_real, x_logit_fake, z_logit_real, z_logit_fake, l_generator, l_encoder)

        loss_gen_enc.backward(retain_graph=True)
        
        optimizer_.step()
        # optimizer_gen.step()
        loss_encoder_s += loss_enc.item()
        loss_decoder_s += loss_gen.item()



    ##############################################

    # else:
        optimizer_dis.zero_grad()
        # optimizer_zz.zero_grad()
        # optimizer_xx.zero_grad()
        # optimizer.zero_grad()
        # x_input = Variable(batch_features.cuda(), requires_grad=True)
        # z_input = Variable(latent_features.cuda(), requires_grad=True)

       
        # x_logit_real, x_logit_fake, z_logit_real, z_logit_fake, l_generator, l_encoder = model(x_input, z_input)

        dis_loss, dis_loss_xz,dis_loss_xx,dis_loss_zz = lossF_dis(x_logit_real, x_logit_fake, z_logit_real, z_logit_fake, l_generator, l_encoder)
        dis_loss.backward()

        # import pdb; pdb.set_trace()

        optimizer_dis.step()
        # optimizer_xx.step()
        # optimizer_zz.step()
        # optimizer_D.step()

        loss_xx_s += dis_loss_xx.item()
        loss_xz_s += dis_loss_xz.item()
        loss_zz_s += dis_loss_zz.item()


        # loss += train_loss.item()

    loss_encoder_s = loss_encoder_s / len(train_loader)
    loss_decoder_s = loss_decoder_s / len(train_loader)
    loss_xx_s = loss_xx_s / len(train_loader)
    loss_xz_s = loss_xz_s / len(train_loader)
    loss_zz_s = loss_zz_s / len(train_loader)

    print("epoch : {}, loss_enc = {:.6f}, loss_dec = {:.6f}, loss_xx = {:.6f}, loss_xz = {:.6f}, loss_zz = {:.6f}".format(epoch + 1, loss_encoder_s, loss_decoder_s, loss_xx_s, loss_xz_s, loss_zz_s))


def testing(test_loader):
    i= 0
    result = np.zeros([test_size, 1])
    global best_p
    for idx, (feature, label) in enumerate(tqdm(test_loader)):
        input_feature = feature[:, :input_dim]
        input_feature = input_feature.view(-1, input_dim).cuda()
        category = feature[:, input_dim]
        #import pdb; pdb.set_trace()
        latent_features = torch.from_numpy(np.random.normal(size=[batch_size, 8])).float()
        with torch.no_grad():
            output = model(input_feature, latent_features, is_training=False)

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

    result = np.where(distances < np.percentile(distances, 80), 0, 1)
    df_from_arr = pd.DataFrame(data=[categorys, result, labels]).T
    cat_error = ((df_from_arr.groupby([0]).sum()[1]-df_from_arr.groupby([0]).sum()[2]).abs()/df_from_arr.groupby([0]).count()[1]).to_numpy()

    # acc = np.sum(result==labels)/test_size
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
    return p,r,f

best_acc=0
for epoch in range(50):
    train(epoch, train_loader=train_loader)
    # print(best_loss
    # if (epoch+1)%2!=0:
    p,r,f = testing(test_loader)
    if p> best_acc:
        best_acc = p
    print('precision:{:.6f}, recall:{:.6f}, f1:{:.6f}'.format(p, r, f))
    scheduler1.step(p)
    scheduler2.step(p)
    epoch += 1

print(best_acc)


