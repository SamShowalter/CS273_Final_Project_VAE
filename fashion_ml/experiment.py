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
from tqdm import tqdm_notebook as tqdm
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

#####################################################################################
# Network Blocks
#####################################################################################

#####################################################################################
# Experiment Object
#####################################################################################

class FashionML:

	def __init__(self,
		model, 
		data_loader, 
		optimizer,
		vae = True,
		val_frac = 0.1,
		learning_rate = 0.001,
		epochs = 100):

		self.vae = vae
		self.model = model 
		self.loader = data_loader
		self.train_size = len(data_loader.train_loader.dataset)
		self.test_size = len(data_loader.test_loader.dataset)
		self.optimizer = optimizer(self.model.parameters(), lr = learning_rate)
		self.epochs = epochs 



	def fit(self, path = "../models/"):

		running_loss = 0
		self.model.cuda()

		for epoch in tqdm(range(self.epochs)):

			for i, data in enumerate(self.loader.train_loader, 0):

				batch, labels = data 
				batch_size = batch.shape[0]

				self.optimizer.zero_grad()

				#Zero because VAE includes lots of data as output
				if self.vae:
					y_hats, batch, mu, logvar  = self.model.forward(batch.cuda())

				loss = self.model.loss_function(y_hats, batch, 
						mu, logvar)
				loss.backward()
				self.optimizer.step()
				running_loss += loss

			#print("Getting validation Loss")
			print('''Validation Loss: {}\nReconstruction Loss: {}\nKLD Loss: {}'''.format(*self.get_val_loss()))
			print("-----"*25)
			#self.plot_sample()

	def plot_sample(self):
		print("Generated Sample: ")
		plt.imshow(torch.squeeze(self.model.sample(1).detach().cpu()), cmap = "Greys");


	def get_val_loss(self, beta = 5):

		val_loss = 0
		val_rec_loss = 0
		val_kld_loss = 0
		with torch.no_grad():
			for i, data in enumerate(self.loader.val_loader):

				batch, labels = data 
				batch_size = batch.shape[0]

				y_hats, batch, mu, logvar = self.model.forward(batch.cuda())

				loss, rec_loss, kld_loss = self.model.loss_function(y_hats, batch, mu, 
					logvar, include_all = True)

				val_loss += loss/128; val_rec_loss += rec_loss/128; val_kld_loss += kld_loss*beta/128;


			return val_loss, val_rec_loss, val_kld_loss










