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

		input_dim = 28*28,
		path = None,
		name = 'FashionVAE'):
		super(FashionVAE, self).__init__()

		self.