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


'''
Two big things
- Size of input
- size of latent dimension 

reconstruction loss function more data driven 
- MSE assumes that it is a normal distribution -- Unnormalized likelihood of normal dist

How to weight total loss
- Paper talks about this : beta VAE - https://openreview.net/forum?id=Sy2fzU9gl
- Beta is set
- Also depends on the dimensionality of things
- KL divergence proportional to latent space
- Minibatch of n should still have equal weight to everything

'''

class FashionVAE(nn.Module):

	def __init__(self,
		in_channels = 1,
		input_dim = 28*28,
		latent_dim = 2,
		h_dims = [32,64,128],
		path = None,
		name = 'FashionVAE',
		):
		super(FashionVAE, self).__init__()
		self.in_channels = in_channels
		self.latent_dim = latent_dim
		self.hidden_dims = h_dims
		self.reconstruction_loss = F.mse_loss
		self.path = path 
		self.name = name

		encoder_blocks = []

		for dim in h_dims:
			encoder_blocks.append(self.__conv_norm_block_encoder(in_channels, dim))
			in_channels = dim

		self.encoder = nn.Sequential(*encoder_blocks)
		#(1,28,28) -> (32,10,10) -> (64,3,3) -> (128,1,1)
		self.fc_mu = nn.Linear(h_dims[-1], latent_dim)
		self.fc_log_var = nn.Linear(h_dims[-1], latent_dim)

		self.decoder_latent_space = nn.Linear(latent_dim, h_dims[-1])

		decoder_blocks = []

		for dim in range(len(h_dims) - 2, -1, -1):
			decoder_blocks.append(self.__conv_norm_block_decoder(h_dims[dim+1], h_dims[dim]))

		self.decoder = nn.Sequential(*decoder_blocks)

		self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(h_dims[0],
                                               h_dims[0],
                                               kernel_size=4,
                                               stride=3,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(h_dims[0]),
                            nn.LeakyReLU(),
                            nn.Conv2d(h_dims[0], out_channels= 1,
                                      kernel_size= 2, padding= 1),
                            nn.Tanh())


	def __conv_norm_block_encoder(self, in_channels, out_channels, kernel_size = 4, stride = 3, padding = 1):
		return nn.Sequential(
			nn.Conv2d(in_channels, out_channels,
				kernel_size = kernel_size, stride = stride, padding = padding),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU())

	def __conv_norm_block_decoder(self, in_channels, out_channels, kernel_size = 4, 
		stride = 3, padding = 1, output_padding = 1):
		return nn.Sequential(
			nn.ConvTranspose2d(in_channels, out_channels,
				kernel_size = kernel_size, stride = stride, 
				padding = padding,
				 output_padding =  padding),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU())


	def reparameterize(self, mu, var):
		std = torch.exp(0.5 * var)
		eps = torch.randn_like(std)
		return eps * std + mu

	def sample(self, n_samples, include_locations = False):

		rand_latent = torch.randn((n_samples, self.latent_dim))

		if include_locations:
			return rand_latent, self.decode(rand_latent.cuda())
		return self.decode(rand_latent.cuda())

	def sample_latent(self, latent):
		return self.decode(torch.tensor(latent).cuda()).detach().cpu()

	def generate(self, x):
		self.forward(x)[0]

	def encode(self, x):
		enc = self.encoder(x)
		enc_flat = torch.flatten(enc, start_dim = 1)
		mu = self.fc_mu(enc_flat)
		log_var = self.fc_log_var(enc_flat)
		var = torch.exp(log_var)

		return [mu, var]

	def decode(self, z):
		dec = self.decoder_latent_space(z)

		dec_reshape = dec.view(-1, self.hidden_dims[-1], 1,1)

		result = self.decoder(dec_reshape)
		# print(result.shape)
		result = self.final_layer(result)

		return result


	def loss_function(self, reconstruction, input_img, mu, var,
		kld_weight):

		reconstruction_loss = self.reconstruction_loss(reconstruction, input_img)
		# print(var)
		kld_loss = 0.5*torch.mean(var + mu**2 - torch.log(var) - 1)
		# print(kld_loss)

		loss = reconstruction_loss + kld_weight*kld_loss
		return loss


	def forward(self, x):
		mu, var = self.encode(x)
		z = self.reparameterize(mu, var)
		# print("HI")
		# print(z.shape)
		# print("Checking")
		r = self.decode(z)
		# print(r.shape)

		return [r, x, mu, var]








		