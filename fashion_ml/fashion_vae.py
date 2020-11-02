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

		blocks = []

		for dim in h_dims:
			blocks.append(self._conv_norm_block(in_channels, dim))
			in_channels = dim

		self.encoder = nn.Sequential(*blocks)
		self.fc_mu = nn.Linear(h_dims[-1]*4, latent_dim)
		self.fc_var = nn.Linear(h_dims[-1]*4)


	def _conv_norm_block(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1):
		return nn.Sequential(
			nn.Conv2d(in_channels, out_channels,
				kernel_size = kernel_size, stride = stride, padding = padding),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU())

		