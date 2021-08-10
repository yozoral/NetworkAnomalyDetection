# -*- coding: utf-8 -*-
# @Author: Jingyi
# @Date:   2020-10-06 20:36:17
# @Last Modified by:   jingyi
# @Last Modified time: 2020-10-06 20:38:19

import pandas as pd

data1 = pd.read_csv("/home/jingyi/Desktop/UNSW-NB15_training.csv")
data2 = pd.read_csv("/home/jingyi/Desktop/UNSW-NB15_valid.csv")

frames = [data1, data2]
result = pd.concat(frames)
result.to_csv("train.csv", index=False)