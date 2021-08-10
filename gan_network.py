# -*- coding: utf-8 -*-
# @Author: Jingyi
# @Date:   2020-08-19 21:56:37
# @Last Modified by:   jingyi
# @Last Modified time: 2020-10-26 22:25:16

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class encoder(nn.Module):
    """docstring for encoder"""
    def __init__(self, input_shape, latent_dim):
        super(encoder, self).__init__()
        self.layer1 = nn.Linear(in_features=input_shape, out_features=1536)
        self.layer2 = nn.Linear(in_features=1536, out_features=1024)
        self.layer3 = nn.Linear(in_features=1024, out_features=512)
        self.layer4 = nn.Linear(in_features=512, out_features=384)
        self.layer5 = nn.Linear(in_features=384, out_features=256)
        self.layer6 = nn.Linear(in_features=256, out_features=128)
        self.layer7 = nn.Linear(in_features=128, out_features=64)
        self.layer8 = nn.Linear(in_features=64, out_features=32)
        self.layer9 = nn.Linear(in_features=32, out_features=16)
        self.layer10 = nn.Linear(in_features=16, out_features=latent_dim)


    def forward(self,encoder_in):
        out = F.leaky_relu(self.layer1(encoder_in))
        out = F.leaky_relu(self.layer2(out))
        out = F.leaky_relu(self.layer3(out))
        out = F.leaky_relu(self.layer4(out))
        out = F.leaky_relu(self.layer5(out))
        out = F.leaky_relu(self.layer6(out))
        out = F.leaky_relu(self.layer7(out))
        out = F.leaky_relu(self.layer8(out))
        out = F.leaky_relu(self.layer9(out))
        out = self.layer10(out)
        return out


class decoder(nn.Module):
    """docstring for decoder"""
    def __init__(self, latent_dim, input_shape):
        super(decoder, self).__init__()
        self.layer1 = nn.Linear(in_features=latent_dim, out_features=16)
        self.layer2 = nn.Linear(in_features=16, out_features=32)
        self.layer3 = nn.Linear(in_features=32, out_features=64)
        self.layer4 = nn.Linear(in_features=64, out_features=128)
        self.layer5 = nn.Linear(in_features=128, out_features=256)
        self.layer6 = nn.Linear(in_features=256, out_features=384)
        self.layer7 = nn.Linear(in_features=384, out_features=512)
        self.layer8 = nn.Linear(in_features=512, out_features=1024)
        self.layer9 = nn.Linear(in_features=1024, out_features=1536)
        self.layer10 = nn.Linear(in_features=1536, out_features=input_shape)

    def forward(self,decoder_in):
        out = F.leaky_relu(self.layer1(decoder_in))
        out = F.leaky_relu(self.layer2(out))
        out = F.leaky_relu(self.layer3(out))
        out = F.leaky_relu(self.layer4(out))
        out = F.leaky_relu(self.layer5(out))
        out = F.leaky_relu(self.layer6(out))
        out = F.leaky_relu(self.layer7(out))
        out = F.leaky_relu(self.layer8(out))
        out = F.leaky_relu(self.layer9(out))
        out = self.layer10(out)
        return out


class discriminator_xz(nn.Module):
    """docstring for discriminator_xz"""
    def __init__(self, x_dim, z_dim):
        super(discriminator_xz, self).__init__()
        self.fc_x = nn.Linear(x_dim, 128)
        self.bn_x = nn.BatchNorm1d(128)
        self.fc_z = nn.Linear(z_dim, 128)
        self.dropout_xz = nn.Dropout(p=0.5)
        self.fc_y = nn.Linear(256, 128) #dim not for sure
        self.fc = nn.Linear(128, 1)

    def forward(self,x, z):
        x = self.fc_x(x)
        x = F.leaky_relu(self.bn_x(x))
        z = F.leaky_relu(self.fc_z(z))
        z = self.dropout_xz(z)
        y = torch.cat((x, z), 1)
        y = self.dropout_xz(F.leaky_relu(self.fc_y(y)))
        logit = self.fc(y)
        return logit, y


class discriminator_xx(nn.Module):
    """docstring for discriminator_xx"""
    def __init__(self, x_dim, x_con):
        super(discriminator_xx, self).__init__()
        self.fc_xx = nn.Linear(x_dim+x_con, 128)
        self.dropout_xx = nn.Dropout(p=0.2)
        self.fc_xx_2 = nn.Linear(128, 64)
        self.fc_xx_3 = nn.Linear(64, 1)

    def forward(self,x, x_con):
        x = torch.cat((x, x_con), 1)
        x = self.dropout_xx(F.leaky_relu(self.fc_xx(x)))
        x = self.fc_xx_2(x)
        logit = self.fc_xx_3(x)
        return logit, x


class discriminator_zz(nn.Module):
    """docstring for discriminator_zz"""
    def __init__(self, z_dim, z_con):
        super(discriminator_zz, self).__init__()
        self.fc_zz = nn.Linear(z_dim+z_con, 64)
        self.fc_zz_2 = nn.Linear(64, 32)
        self.fc_zz_3 = nn.Linear(32, 1)
        self.dropout_zz = nn.Dropout(p=0.2)

    def forward(self,z, z_con):
        z = torch.cat((z, z_con), -1)
        zz = z.clone()
        zz = self.dropout_zz(F.leaky_relu(self.fc_zz(zz)))
        zz = self.fc_zz_2(zz)
        zz = F.leaky_relu(zz)
        zz = self.dropout_zz(zz)
        # z = self.dropout_zz(F.leaky_relu(self.fc_zz_2(z.clone())))
        logit = self.fc_zz_3(zz)
        return logit, zz


class ALAD(nn.Module):
    """docstring for ALAD"""
    def __init__(self, input_dim, latent_dim):
        super(ALAD, self).__init__()
        self.enc = encoder(input_dim, latent_dim)
        self.gen = decoder(latent_dim, input_dim)
        self.dis_xx = discriminator_xx(input_dim, input_dim)
        self.dis_zz = discriminator_zz(latent_dim, latent_dim)
        self.dis_xz = discriminator_xz(input_dim, latent_dim)
    

    def forward(self, x_input, z_input, is_training=True):
        if is_training:
            z_gen = self.enc(x_input)
            x_gen = self.gen(z_input)
            recon_x = self.gen(z_gen)
            recon_z = self.enc(x_gen)

            # with torch.no_grad():
            l_encoder, inter_layer_inp_xz = self.dis_xz(x_input, z_gen)
            l_generator, inter_layer_rct_xz = self.dis_xz(x_gen, z_input)

            x_logit_real, inter_layer_inp_xx = self.dis_xx(x_input, x_input)
            x_logit_fake, inter_layer_rct_xx = self.dis_xx(x_input, recon_x)

            z_logit_real, _ = self.dis_zz(z_input, z_input)
            z_logit_fake, _ = self.dis_zz(z_input, recon_z)

            # l_generator = Variable(l_generator, requires_grad=True)
            # l_encoder = Variable(l_encoder, requires_grad=True)
            # x_logit_real = Variable(x_logit_real, requires_grad=True)
            # x_logit_fake = Variable(x_logit_fake, requires_grad=True)
            # z_logit_real = Variable(z_logit_real, requires_grad=True)
            # z_logit_fake = Variable(z_logit_fake, requires_grad=True)
            return x_logit_real, x_logit_fake, z_logit_real, z_logit_fake, l_generator, l_encoder
        else:
            z_gen = self.enc(x_input)
            recon_x = self.gen(z_gen)
            return recon_x