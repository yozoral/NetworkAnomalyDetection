# -*- coding: utf-8 -*-
# @Author: Jingyi
# @Date:   2020-07-09 22:59:13
# @Last Modified by:   jingyi
# @Last Modified time: 2021-01-12 20:49:34
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing


'''
class unsw(Dataset):
    def __init__(self, csv_file):
        super(unsw, self).__init__()
        data = pd.read_csv(csv_file)
        data = data.replace(r'\s+', np.nan, regex=True)
        data = data.fillna(0)
        self.label = data['Label'].to_numpy(dtype=np.int)
        self.data = data.drop(columns=['Label','attack_cat']).to_numpy(dtype=np.float32)



        import pdb; pdb.set_trace()

        # self.feature_list = pd.read_csv("./unsw/UNSW-NB15_features.csv")['Name']
        # self.data.columns = self.feature_list

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        # import pdb; pdb.set_trace()
        label = self.label[idx, :]
        x = self.data[idx, :]
        import pdb; pdb.set_trace()
        x = torch.tensor(x.values.astype(np.float))
        label = torch.tensor(label)
        return x, label
'''

def unsw(csv_file, is_train):
    data = pd.read_csv(csv_file)
    if is_train!=True:
        # import pdb; pdb.set_trace()
        data['attack_cat'], unique = pd.factorize(data['attack_cat'])
        print(unique)

    label = data['label'].to_numpy(dtype=np.int)
    data = data.drop(columns=['label']).to_numpy(dtype=np.float32)
    # data = data.drop(columns=['Label','attack_cat']).to_numpy(dtype=np.float32)
    data = torch.from_numpy(data)
    label = torch.from_numpy(label)
    return data, label

def cicids(csv_file, is_train):
    data = pd.read_csv(csv_file)
    if is_train!=True:
        # import pdb; pdb.set_trace()
        cols_to_order = ['attack_cat']
        new_columns = (data.columns.drop(cols_to_order).tolist()) + cols_to_order
        data = data[new_columns]
        data['attack_cat'], unique = pd.factorize(data['attack_cat'])
        print(unique)
        label = data['label'].to_numpy(dtype=np.int)
        data = data.drop(columns=['label']).to_numpy(dtype=np.float32)
        # data = data.drop(columns=['Label','attack_cat']).to_numpy(dtype=np.float32)
    else:
        label = data['label'].to_numpy(dtype=np.int)
        data = data.drop(columns=['label', 'attack_cat']).to_numpy(dtype=np.float32)
        # data = data.drop(columns=['Label','attack_cat']).to_numpy(dtype=np.float32)
    data = torch.from_numpy(data)
    label = torch.from_numpy(label)
    return data, label