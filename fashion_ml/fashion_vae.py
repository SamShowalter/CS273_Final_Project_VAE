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
import sys

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
		latent_dim = 3,
		h_dims = [32,32,64],
		strides = [1,2,2],
		padding = [2,1,1],
		path = None,
		name = 'FashionVAE',
		):
		'''
		(n_samples, 1, 28 ,28)
		(n_samples, 32,14,14)
		(n_samples, 32,14,14)
		(n_sampes, 64,7,7)

		'''
		super(FashionVAE, self).__init__()
		self.in_channels = in_channels
		self.latent_dim = latent_dim
		self.hidden_dims = h_dims
		self.reconstruction_loss = self.__MSE
		self.path = path 
		self.strides = strides
		self.padding = padding
		self.name = name

		encoder_blocks = []

		for dim, stride, pad in zip(h_dims,strides,padding):
			encoder_blocks.append(self.__conv_norm_block_encoder(in_channels, dim, 
				stride = stride, padding = pad))
			in_channels = dim

		self.encoder = nn.Sequential(*encoder_blocks)
		#(1,28,28) -> (32,10,10) -> (64,3,3) -> (128,1,1)
		self.fc_mu = nn.Linear(h_dims[-1]*7*7, latent_dim)
		self.fc_log_var = nn.Linear(h_dims[-1]*7*7, latent_dim)

		self.decoder_latent_space = nn.Sequential(
			nn.Linear(latent_dim, h_dims[-1]*7*7),
			nn.LeakyReLU())

		decoder_blocks = []

		t_hdims = [self.in_channels] + h_dims
		for dim in range(len(t_hdims)-1, 0, -1):
			decoder_blocks.append(self.__conv_norm_block_decoder(t_hdims[dim], t_hdims[dim-1],
				stride = strides[dim-1], padding = padding[dim-1], 
				output_padding = 1 if t_hdims[dim] == t_hdims[dim-1] else 0))
			if dim != 1:
				decoder_blocks.append(nn.LeakyReLU())

		self.decoder = nn.Sequential(*decoder_blocks)


	def __conv_norm_block_encoder(self, in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1):
		return nn.Sequential(
			nn.Conv2d(in_channels, out_channels,
				kernel_size = kernel_size, stride = stride, padding = padding),
			nn.LeakyReLU())

	def __conv_norm_block_decoder(self, in_channels, out_channels, kernel_size = 4, 
		stride = 2, padding = 1, output_padding = 0):
		return nn.ConvTranspose2d(in_channels, out_channels,
				kernel_size = kernel_size, stride = stride, 
				padding = padding,
				 output_padding =  output_padding)

	def sample(self, n_samples, include_locations = False):

		rand_latent = torch.randn((n_samples, self.latent_dim))

		if include_locations:
			return rand_latent, self.decode(rand_latent.cuda())
		return self.decode(rand_latent.cuda())

	def sample_latent(self, latent):
		return self.decode(torch.tensor(latent).cuda()).detach().cpu()

	def generate(self, x):
		self.forward(x)[0]


	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return eps * std + mu

	

	def encode(self, x):
		enc = self.encoder(x)
		enc_flat = torch.flatten(enc, start_dim = 1)
		mu = self.fc_mu(enc_flat)
		logvar = self.fc_log_var(enc_flat)

		return [mu, logvar]

	def decode(self, z):
		dec = self.decoder_latent_space(z)

		dec_reshape = dec.view(-1, self.hidden_dims[-1],7,7)

		result = self.decoder(dec_reshape)

		return result


	def __MSE(self, input_img, reconstruction, dim = 0):

		return torch.mean(
			torch.squeeze(((input_img - reconstruction)**2).sum(axis = 3).sum(axis = 2))
			, axis = 0)

	def interpolate(self,img1,img2,num_samples = 5):

		mu1, logvar1 = self.encode(img1.cuda())
		loc1 = self.reparameterize(mu1, logvar1)

		mu2, logvar2 = self.encode(img2.cuda())
		loc2 = self.reparameterize(mu2, logvar2)
		
		diff = loc2 - loc1

		inter_samples = []

		with torch.no_grad():
			for i in range(num_samples):
				latent = loc1 + (diff / num_samples)*i
				print(latent)

				inter_samples.append(self.decode(latent).cpu())

		return inter_samples



	def loss_function(self, reconstruction, input_img, mu, logvar, beta = 5, include_all = False):

		var = torch.exp(logvar)
		reconstruction_loss = self.reconstruction_loss( input_img, reconstruction)

		kld_loss = torch.mean(0.5* torch.sum(var + mu**2 - torch.log(var) - 1, dim = 1))


		loss = reconstruction_loss + kld_loss*beta
		if include_all:
			return loss, reconstruction_loss, kld_loss

		return loss


	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)

		r = self.decode(z)

		return [r, x, mu, logvar]







		