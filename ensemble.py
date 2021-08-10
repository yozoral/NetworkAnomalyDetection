# -*- coding: utf-8 -*-
# @Author: Jingyi
# @Date:   2020-07-18 01:34:04
# @Last Modified by:   jingyi
# @Last Modified time: 2021-03-12 02:29:19

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ae import AE
from vae import VAE
from cnnae import AutoEncoder
from gan_network import ALAD
from cnnae import AutoEncoder
from cnnvae import VAutoEncoder
from dataset import unsw, cicids
from evaluations import do_prc, do_roc

from tqdm import tqdm
from fastloader import FastTensorDataLoader
from sklearn.metrics.pairwise import paired_distances
from mahalanobis import MahalanobisLayer
import numpy as np
from numpy.linalg import norm
from scipy.optimize import differential_evolution
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
# import matplotlib.pyplot as plt

# SEED =42
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)

batch_size = 512
# test_x, test_y = unsw("/home/jingyi/Avada/cicids2017/MonWedFri_std_norm/cicids2017_test.csv")


val_x, val_y = unsw("/home/jingyi/Avada/unsw/val.csv", is_train=False)
test_x, test_y = unsw("/home/jingyi/Avada/unsw/test.csv", is_train=False)


'''
train_x, train_y = cicids("/home/jingyi/Avada/cicids/cicids2017_rs1_training.csv", is_train=False)
val_x, val_y = cicids("/home/jingyi/Avada/cicids/val.csv", is_train=False)
test_x, test_y = cicids("/home/jingyi/Avada/cicids/test.csv", is_train=False)
# test_x, test_y = cicids("/home/jingyi/Avada/cicids/cicids2017_rs1_test.csv", is_train=False)
'''

# train_loader = FastTensorDataLoader(train_x, train_y, batch_size=batch_size, shuffle=True)
val_loader = FastTensorDataLoader(val_x, val_y, batch_size=batch_size, shuffle=True)
test_loader = FastTensorDataLoader(test_x, test_y, batch_size=batch_size, shuffle=True)
# test_size = test_x.size(0)


feature_number = 41


modelA = AE(input_shape=feature_number).cuda()
modelB = VAE(feature_dim=feature_number).cuda()
modelC = AutoEncoder(feature_number).cuda()
modelD = VAutoEncoder(feature_number).cuda()
modelE = ALAD(input_dim=feature_number, latent_dim=8).cuda()


checkpoint_A=torch.load("unsw_ae.pth.tar")
modelA.load_state_dict(checkpoint_A["model_state_dict"])

checkpoint_B=torch.load("unsw_vae.pth.tar")
modelB.load_state_dict(checkpoint_B["model_state_dict"])

checkpoint_C=torch.load("unsw_cnnae.pth.tar")
modelC.load_state_dict(checkpoint_C["model_state_dict"])

checkpoint_D=torch.load("unsw_cnnvae.pth.tar")
modelD.load_state_dict(checkpoint_D["model_state_dict"])

checkpoint_E=torch.load("unsw_gan.pth.tar")
modelE.load_state_dict(checkpoint_E["model_state_dict"])

'''
checkpoint_A=torch.load("cicids_ae.pth.tar")
modelA.load_state_dict(checkpoint_A["model_state_dict"])

checkpoint_B=torch.load("cicids_vae.pth.tar")
modelB.load_state_dict(checkpoint_B["model_state_dict"])

checkpoint_C=torch.load("cicids_cnnae.pth.tar")
modelC.load_state_dict(checkpoint_C["model_state_dict"])

# import pdb
# pdb.set_trace()

checkpoint_D=torch.load("cicids_cnnvae.pth.tar")
modelD.load_state_dict(checkpoint_D["model_state_dict"])

checkpoint_E=torch.load("cicids_gan.pth.tar")
modelE.load_state_dict(checkpoint_E["model_state_dict"])

'''

n_member = 4
percentile_accuracy = 89

mahalanobis_cov_decay=0.001
mahalanobis_layer = MahalanobisLayer(feature_number, mahalanobis_cov_decay).cuda()

# normalize a weight vector to have unit norm
def normalize(tmp_weights):
    # calculate l1 vector norm
    result_w = norm(tmp_weights, 1)
    # check for a vector of all zeros
    if result_w == 0.0:
        return tmp_weights
    # return normalized vector (unit norm)
    return tmp_weights / result_w

all_models = [modelB, modelC, modelD, modelE]
#, modelC, modelD, modelE

scaler = MinMaxScaler()
# scaler.fit(train_data)
def singleModel(test_loader, model_name):
    i=0
    # import pdb; pdb.set_trace()
    if model_name=='ae':
        model=modelA
    elif model_name=='vae':
        model=modelB
    elif model_name=='cnnae':
        model=modelC
    elif model_name=='cnnvae':
        model=modelD
    elif model_name=='gan':
        model=modelE
    for idx, (feature, label) in enumerate(tqdm(test_loader)):
        input_feature = feature[:, :feature_number]
        category = feature[:, feature_number]
        input_feature = input_feature.cuda()
        latent_features = torch.from_numpy(np.random.normal(size=[batch_size, 8])).float()
        # input_feature = feature.view(-1, 1, 41)
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            if model_name=='gan':
                output = model(input_feature, latent_features, is_training=False)
                # distance = mahalanobis_layer(input_feature, output)
            else:
                output = model(input_feature)
                # distance = mahalanobis_layer(input_feature, output)
        output = output.view(-1, feature_number)
        if i==0:
            outputs = output
            features = input_feature
            labels = label
            categorys = category
            # distances = distance
        else:
            outputs = torch.cat((outputs, output), 0)
            features = torch.cat((features, input_feature), 0)
            labels = torch.cat((labels, label), 0)
            categorys = torch.cat((categorys, category), 0)
            # distances = torch.cat((distances, distance), 0)

        i+=1

    outputs = outputs.cpu().detach().numpy()
    features = features.cpu().detach().numpy()
    labels = labels.numpy()
    categorys = categorys.numpy()
    test_size = labels.shape[0]
    # distances = distances.cpu().detach().numpy()
    distances = paired_distances(features, outputs)

    result = np.where(distances < np.percentile(distances, percentile_accuracy), 0, 1)
    '''
    # import pdb; pdb.set_trace()
    df_from_arr = pd.DataFrame(data=[categorys, result, labels]).T
    cat_error = ((df_from_arr.groupby([0]).sum()[1]-df_from_arr.groupby([0]).sum()[2]).abs()/df_from_arr.groupby([0]).count()[1]).to_numpy()

    acc = np.sum(result==labels)/test_size
    # import pdb; pdb.set_trace()
    p, r, f, _ = precision_recall_fscore_support(labels, result, average='macro')
    '''
    # plt(distances, labels, model_name)
    # np.save("unsw-gan.npy", distances)
    acc = np.sum(result==labels)/test_size
    # import pdb; pdb.set_trace()
    p, r, f, _ = precision_recall_fscore_support(labels, result, average='macro')
    print(model_name, acc, p, r, f)


def plt(test_score, labels, model_name):
    # test_score=result
    # test_score = scaler.fit_transform(result)
    do_prc(test_score, labels, file_name=model_name)
    do_roc(test_score, labels, file_name=model_name)


def testing(loader):
    i=0

    all_outputs=[]
    distances=[]
    for idx, (feature, label) in enumerate(tqdm(loader)):
        # feature = feature.cuda()
        # import pdb; pdb.set_trace()
        input_feature = feature[:, :feature_number].view(-1, feature_number)
        category = feature[:, feature_number]
        input_feature = input_feature.cuda()

        latent_features = torch.from_numpy(np.random.normal(size=[batch_size, 8])).float()
        # import pdb; pdb.set_trace()
        for x in range(n_member):
            model = all_models[x]
            # import pdb; pdb.set_trace()
            if model == modelE:
                with torch.no_grad():
                    output = model(input_feature, latent_features, is_training=False)
                    distance = mahalanobis_layer(input_feature, output)

                    # .view(-1, feature_number)
            else:
                with torch.no_grad():
                    output = model(input_feature)
                    distance = mahalanobis_layer(input_feature, output)

                    # .view(-1, feature_number)
            if i==0:
                all_outputs.append(output)
                distances.append(distance)
            else:
                all_outputs[x] = torch.cat((all_outputs[x], output), 0)
                distances[x] = torch.cat((distances[x], distance), 0)


        if i==0:
            all_outputs.extend((input_feature, label, category))
        else:
            all_outputs[-3] = torch.cat((all_outputs[-3], input_feature), 0)
            all_outputs[-2] = torch.cat((all_outputs[-2], label), 0)
            all_outputs[-1] = torch.cat((all_outputs[-1], category), 0)

        i+=1

    # if loader==test_loader:
    #     import pdb; pdb.set_trace()
    return [all_outputs, distances]

def loss_function(l_weights,l_feature_hat):
    # normalize weights
    normalized = normalize(l_weights)
    # calculate error rate
    return 1.0 - ensemble_prediction(l_feature_hat,normalized)[0][1]


distances_list=[]
def ensemble_prediction(feature_hat, ep_weights, is_best=False):
    model_name='ensemble'

    feature_hats, distances = feature_hat
    # mweights = torch.tensor(np.array(ep_weights)).cuda()
    mweights = np.array(ep_weights)
    myhats = 0

    # import pdb; pdb.set_trace()
    features = feature_hats[-3].cpu().detach().numpy()
    labels = feature_hats[-2].numpy()
    # distances = distances.cpu().detach().numpy()
    #.reshape(-1, 1) if scaler then use reshape

    for x in range(n_member):
        # dis = paired_distances(features, feature_hats[x].cpu().detach())
        dis = distances[x]
        # import pdb; pdb.set_trace()
        # normalized_dis = scaler.transform(dis.reshape(-1, 1))
        myhats += dis * mweights[x]

        '''
        if is_best==False:
            # if x==4:
            # np.clip(dis, 0, 140)
            plt.scatter(range(len(dis)), dis, color='lightblue')
            plt.savefig(str(x)+'_cicids_plt.eps', bbox_inches='tight', format='eps')
        '''

    # for x in range(n_member):
    #     myhats += feature_hats[x] * mweights[x]

    myhats = myhats.cpu().detach().numpy()
    # features = feature_hats[-3].cpu().detach().numpy()
    # labels = feature_hats[-2].numpy()
    category = feature_hats[-1].numpy()

    # distances = paired_distances(features, myhats)

    # quit()
    test_size = labels.shape[0]
    p_result = np.where(myhats < np.percentile(myhats, percentile_accuracy), 0, 1)
    # p_result = np.where(myhats < 0.5, 0, 1)
    # import pdb; pdb.set_trace()
    acc = np.sum(p_result == labels) / test_size
    p, r, f, _ = precision_recall_fscore_support(labels, p_result, average='macro')

    
    if is_best==True:

        df_from_arr = pd.DataFrame(data=[category,p_result, labels]).T
        # import pdb; pdb.set_trace()
        cat_error = ((df_from_arr.groupby([0]).sum()[1]-df_from_arr.groupby([0]).sum()[2]).abs()/df_from_arr.groupby([0]).count()[1]).to_numpy()
        # print(cat_error)
        # print(df_from_arr.groupby([0]).count()[1])
        # distances_list.append(distances)
        # plt(myhats, labels, "all")
        # print(df_from_arr.groupby([0]).sum()[2])

    else:
        cat_error=[]

    # for x in range(n_member):
    #     distances_ae = paired_distances(features, feature_hats[x].cpu().detach().numpy())
    #     ae_result = np.where(distances_ae < np.percentile(distances_ae, percentile_accuracy), 0, 1)
    #     p, r, f, _ = precision_recall_fscore_support(labels, ae_result, average='macro')
    #     print(p, r,f)

    # quit()

    
    
    return [acc, p, r, f], cat_error

'''
singleModel(test_loader, model_name="gan")
singleModel(test_loader, model_name="ae")
singleModel(test_loader, model_name="vae")
singleModel(test_loader, model_name="cnnae")
singleModel(test_loader, model_name="cnnvae")
'''

weights = [1.0 / n_member for _ in range(n_member)]
eval_result = testing(val_loader)

[score, p, r, f], c = ensemble_prediction(eval_result,  weights)
print('Equal Weights acc:{:.6f}, precision:{:.6f}, recall:{:.6f}, f1:{:.6f}'.format(score, p, r, f))

# define bounds on each weight
bound_w = [(0.0, 1.0) for _ in range(n_member)]
# arguments to the loss function
search_arg = (eval_result,)
# global optimization of ensemble weights
result = differential_evolution(loss_function, bound_w, search_arg, maxiter=2, tol=1e-7)
found_weights = normalize(result['x'])
print('Optimized found weights for ensemble models: %s' % found_weights)

test_result = testing(test_loader)
[score, p, r, f], cat_error = ensemble_prediction(test_result, found_weights, is_best=True)
print('Optimized Weights acc:{:.6f}, precision:{:.6f}, recall:{:.6f}, f1:{:.6f}'.format(score, p, r, f))
# print(cat_error)
