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

from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import torch
import numpy as np
import pickle as pkl
import copy

#####################################################################################
# Data transforms for all models
#####################################################################################


class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)

rotation_transform = RotationTransform(angle=90)

################## Densenet, Wide Resnet ##################

#TODO: Figure out if FashionMNIST needs normalization
normalize = None

transform_train = transforms.Compose([
        transforms.ToTensor()
        # normalize
      
    ])

transform_test = transforms.Compose([
        transforms.ToTensor()
        # normalize
    ])


#####################################################################################
# Dataloader object
#####################################################################################

class EZ_Dataloader:

	def __init__(self,  train_source,
						transform_train = transform_train,
						transform_test = transform_test,
						batch_size = 1,
						root_storage = "../"):

		self.batch_size = batch_size
		self.root_storage = root_storage
		self.train_source = train_source
		self.train_loader = None 
		self.test_loader = None

		self.transform_train = transform_train
		self.transform_test = transform_test

		self.sources = {
						"CIFAR10":{"source": datasets.CIFAR10,
								   "path":'data'},
						"CIFAR100":{"source": datasets.CIFAR100,
								   "path":'data'},
						"MNIST":{"source": datasets.MNIST,
								   "path":'data'},
						"KMNIST":{"source": datasets.KMNIST,
								   "path":'data',
								   'split':'test'},
						"FashionMNIST":{"source": datasets.FashionMNIST,
								   "path":'data'}
					   }

		assert train_source in self.sources.keys(), "ERROR: Train source \"{}\" not found in repository.\nCheck spelling and try again".format(train_source) 
		
	def __load_pickle_file(self, path):

		with open(path, "rb") as file:
			return pkl.load(file)

		
	def __build_data_loader(self, source, batch_size = 1, train = False, transform_train = True, shuffle = False):

		loader = None
		source_dict = self.sources[source]


		if source_dict.get("split",None) is not None:

			loader = torch.utils.data.DataLoader(
									source_dict["source"](
										self.root_storage + source_dict["path"], 
										split = source_dict["split"],
										download = True,
										transform = self.transform_train if transform_train else self.transform_test),
									batch_size = batch_size,
									shuffle = shuffle,
								)

		else: 

			loader = torch.utils.data.DataLoader(
									source_dict["source"](
										self.root_storage + source_dict["path"], 
										download = True,
										train = train,
										transform = self.transform_train if transform_train else self.transform_test),
									batch_size = batch_size,
									shuffle = shuffle,
								)
		return loader 

	def build_data_loader(self, data, batch_size = 128, shuffle = False):

		return torch.utils.data.DataLoader(
									data,
									batch_size = batch_size,
									shuffle = shuffle,
								)

	def build_val_loader(self, frac = 0.1):

		batch_size = self.train_loader.batch_size

		data = self.train_loader.dataset 
		lens = [int((1-frac)*len(data)), int(frac*len(data))]
		train_data, val_data = torch.utils.data.dataset.random_split(data, lens)

		self.train_loader = self.build_data_loader(train_data, batch_size = batch_size)
		self.val_loader = self.build_data_loader(val_data, batch_size = batch_size)


	def build_train_test_loader(self):

		self.train_loader = self.__build_data_loader(self.train_source,
													batch_size = 128, 
													train = True, 
													transform_train = True,
													shuffle = False)

		self.test_loader = self.__build_data_loader(self.train_source,
													batch_size = 128, 
													train = False, 
													transform_train = False,
													shuffle = False)


	def reset_train_loader(self, batch_size = 1, shuffle = False, train = False, transform_train = True):

	    self.train_loader = self.__build_data_loader(self.train_source,
	                                                batch_size, 
	                                                shuffle = shuffle,
	                                                transform_train = transform_train,
	                                                train = train)

	def reset_test_loader(self, batch_size = 1, shuffle = False, train = False, transform_train = False):

	    self.test_loader = self.__build_data_loader(self.train_source,
	                                                batch_size,
	                                                shuffle = shuffle, 
	                                                transform_train = transform_train,
	                                                train = False)
								

	

