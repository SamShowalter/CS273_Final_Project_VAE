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
		learning_rate = 0.0001,
		epochs = 100):

		self.vae = vae
		self.model = model 
		self.loader = data_loader
		self.train_size = len(data_loader.train_loader.dataset)
		self.test_size = len(data_loader.test_loader.dataset)
		self.optimizer = optimizer(self.model.parameters(), lr = learning_rate)
		self.epochs = epochs 



	def fit(self, path = "../models/"):

		total_samples = len(self.loader.train_loader.dataset)
		running_loss = 0
		self.model.cuda()

		for epoch in tqdm(range(self.epochs)):

			for i, data in enumerate(self.loader.train_loader, 0):

				batch, labels = data 
				batch_size = batch.shape[0]

				self.optimizer.zero_grad()

				#Zero because VAE includes lots of data as output
				if self.vae:
					y_hats, batch, mu, var,  = self.model.forward(batch.cuda())

				loss = self.model.loss_function(y_hats, batch, 
						mu, var, batch_size/total_samples)
				loss.backward()
				self.optimizer.step()
				running_loss += loss
			
				# if i % 100 == 99:    # print every 2000 mini-batches
				# 	print('[{},{}] loss: {}'.format(epoch + 1, i + 1, running_loss))
				# 	running_loss = 0

			#print("Getting validation Loss")
			print("Validation Loss: {}".format(self.get_val_loss()))
			#self.plot_sample()

	def plot_sample(self):
		print("Generated Sample: ")
		plt.imshow(torch.squeeze(self.model.sample(1).detach().cpu()), cmap = "Greys");


	def get_val_loss(self):

		val_loss = 0
		total_samples = len(self.loader.train_loader.dataset)
		with torch.no_grad():
			for i, data in enumerate(self.loader.val_loader):

				batch, labels = data 
				batch_size = batch.shape[0]

				y_hats, batch, mu, var = self.model.forward(batch.cuda())

				val_loss += self.model.loss_function(y_hats, batch, mu, var, batch_size/total_samples)

			return val_loss










