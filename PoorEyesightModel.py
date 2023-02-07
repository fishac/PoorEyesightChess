import torch
import torch.nn as nn
import torch.nn.functional as F
import SquareData

class PoorEyesightSquareModel(nn.Module):
	def __init__(self, square, layer1_size, layer2_size, layer3_size, layer4_size):
		super().__init__()
		self.square = square
		self.layer1_size = layer1_size
		self.layer2_size = layer2_size
		self.layer3_size = layer3_size
		self.layer4_size = layer4_size
		self.layer1 = nn.Linear(64*2, layer1_size)
		self.layer2 = nn.Linear(layer1_size, layer2_size)
		self.layer3 = nn.Linear(layer2_size, layer3_size)
		self.layer3 = nn.Linear(layer3_size, layer4_size)
		self.output_layer = nn.Linear(layer4_size, SquareData.square_total_occupants[square])
		self.nonlinearity = nn.ReLU()
		
	def forward(self, x):
		x = self.nonlinearity(self.layer1(x))
		x = self.nonlinearity(self.layer2(x))
		x = self.nonlinearity(self.layer3(x))
		x = F.softmax(self.output_layer(x),dim=1)
		return x
	
