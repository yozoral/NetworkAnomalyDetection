# -*- coding: utf-8 -*-
# @Author: Jingyi
# @Date:   2020-09-17 22:09:23
# @Last Modified by:   jingyi
# @Last Modified time: 2020-09-17 23:01:22

import torch
import torch.nn as nn

class encLoss(nn.Module):
    """docstring for aladLoss"""
    def __init__(self):
        super(encLoss, self).__init__()
        self.criterion= nn.BCEWithLogitsLoss()

    def forward(self, x_logit_real, x_logit_fake, z_logit_real, z_logit_fake, l_generator, l_encoder):
        
        gen_loss_xz = torch.mean(self.criterion(l_generator, torch.ones_like(l_generator)))
        enc_loss_xz = torch.mean(self.criterion(l_encoder, torch.zeros_like(l_encoder)))
        x_real_gen = self.criterion(x_logit_real, torch.zeros_like(x_logit_real))
        x_fake_gen = self.criterion(x_logit_fake, torch.ones_like(x_logit_fake))
        z_real_gen = self.criterion(z_logit_real, torch.zeros_like(z_logit_real))
        z_fake_gen = self.criterion(z_logit_fake, torch.ones_like(z_logit_fake))

        cost_x = torch.mean(x_real_gen + x_fake_gen)
        cost_z = torch.mean(z_real_gen + z_fake_gen)

        cycle_consistant_loss = cost_x + cost_z

        loss_generator = gen_loss_xz + cycle_consistant_loss
        loss_encoder = enc_loss_xz + cycle_consistant_loss
        loss_gen_enc = enc_loss_xz + gen_loss_xz + cycle_consistant_loss
        return loss_gen_enc, loss_encoder, loss_generator

class discriLoss(nn.Module):
    """docstring for aladLoss"""
    def __init__(self):
        super(discriLoss, self).__init__()
        self.criterion= nn.BCEWithLogitsLoss()

    def forward(self, x_logit_real, x_logit_fake, z_logit_real, z_logit_fake, l_generator, l_encoder):
        
        loss_dis_enc = torch.mean(self.criterion(l_encoder, torch.ones_like(l_encoder)))
        loss_dis_gen = torch.mean(self.criterion(l_generator, torch.zeros_like(l_generator)))
        dis_loss_xz = loss_dis_gen + loss_dis_enc

        x_real_dis = self.criterion(x_logit_real, torch.ones_like(x_logit_real))
        x_fake_dis = self.criterion(x_logit_fake, torch.zeros_like(x_logit_fake))
        dis_loss_xx = torch.mean(x_real_dis + x_fake_dis)

        z_real_dis = self.criterion(z_logit_real, torch.ones_like(z_logit_real))
        z_fake_dis = self.criterion(z_logit_fake, torch.zeros_like(z_logit_fake))
        dis_loss_zz = torch.mean(z_real_dis + z_fake_dis)

        return dis_loss_zz+dis_loss_xx+dis_loss_xz, dis_loss_xz, dis_loss_xx, dis_loss_zz