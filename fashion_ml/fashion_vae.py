#####################################################################################
#
#
# 	Dataset loader for Fashion MNIST Final Project
#	-- Fall 2020 - CS 273 - Machine Learning --
#  
#	Author: Sam Showalter
#	Date: November 1, 2020
#
#####################################################################################


#####################################################################################
# External Library and Module Imports
#####################################################################################

#Math libraries
import random
import numpy as np

#Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable, grad
from torchvision import datasets, transforms
from torch.nn.parameter import Parameter

#####################################################################################
# Network Blocks
#####################################################################################

#####################################################################################
# VAE Network
#####################################################################################

class FashionVAE(nn.Module):

	def __init__(self,
		in_channels = 3,
		input_dim = 28*28,
		latent_dim = 2,
		h_dims = [32,64,128,256],
		path = None,
		name = 'FashionVAE',
		):
		super(FashionVAE, self).__init__()
		self.in_channels = in_channels
		self.latent_dim = latent_dim
		self.hidden_dims = h_dims
		self.path = path 
		self.name = name

		encoder_blocks = []

		for dim in h_dims:
			blocks.append(self._conv_norm_block_encoder(in_channels, dim))
			in_channels = dim

		self.encoder = nn.Sequential(*encoder_blocks)
		self.fc_mu = nn.Linear(h_dims[-1]*4, latent_dim)
		self.fc_var = nn.Linear(h_dims[-1]*4, latent_dim)

		self.decoder_latent_space = nn.Linear(latent_dim, hiddem_dims[-1]*4)

		decoder_blocks = []

		for dim in range(len(h_dims) - 1, -1, -1):
			decoder_blocks.append(self._conv_norm_block_decoder(h_dims[i+1], h_dims[i]))


		self.decoder = nn.Sequential(*decoder_blocks)

		self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(h_dims[-1],
                                               h_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(h_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(h_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())


	def _conv_norm_block_encoder(self, in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1):
		return nn.Sequential(
			nn.Conv2d(in_channels, out_channels,
				kernel_size = kernel_size, stride = stride, padding = padding),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU())

	def _conv_norm_block_decoder(self, in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1, ouput_padding = 1):
		return nn.Sequential(
			nn.ConvTranspose2d(in_channels, out_channels,
				kernel_size = kernel_size, stride = stride, 
				padding = padding, output_padding = output_padding),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU())

	def encode(self, x):

		enc = self.encoder(x)
		enc_flat = torch.flatten(enc, start_dim = 1)
		mu = self.fc_mu(enc_flat)
		var = self.fc_var(enc_flat)

		return [mu, var]

	def decode(self, z):
		dec = self.decoder_latent_space(z)

		#WHY IS THIS (2,2) at end
		dec_reshape = dec.view(-1, h_dim[-1], 2, 2)

		result = self.decoder(dec_reshape)
		result = self.final_layer(result)

		return result






		