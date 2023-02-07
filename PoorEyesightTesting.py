import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time
import SquareData
import PoorEyesightModel
from math import ceil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_epochs = 5
batch_size = 1000
total_files_train = 55
total_files_test = 10
criterion = nn.CrossEntropyLoss()

def eval(model,square,test_data_df):
	model.eval()
	model.to('cpu')
	print(f'Square performance: {square}',flush=True)
	for i,occupant in enumerate(SquareData.occupants):
		total_correct = 0
		data_idx = 0
		total_tests = len(test_data_df[test_data_df[square + '_' + occupant]==1])
		n_test_batches = ceil(total_tests/5000)
		for j in range(n_test_batches):
			test_input = torch.FloatTensor(test_data_df[test_data_df[square + '_' + occupant]==1][SquareData.input_features].values)[data_idx:min([(data_idx+5000),total_tests])]
			test_output = torch.FloatTensor(test_data_df[test_data_df[square + '_' + occupant]==1][SquareData.output_features_dict[square]].values)[data_idx:min([(data_idx+5000),total_tests])]
			test_out_prob = model(test_input)
			test_out_pred = torch.argmax(test_out_prob,dim=1)
			test_out_actual = torch.argmax(test_output,dim=1)
			total_correct += sum(test_out_pred == test_out_actual)
			data_idx += 5000
		print(f'\tOccupant: {occupant}, total in test: {total_tests}, total correct: {total_correct}, proportion: {total_correct/total_tests:.6f}',flush=True)
	model.to(device)

def main():
	test_data_dfs = []
	for i in range(total_files_test):
		test_data_df = pd.read_csv(f'./processed_data/processed_data_{i+total_files_train}.csv')
		test_data_dfs.append(test_data_df)
	test_data_df = pd.concat(test_data_dfs,axis=0)
	
	square = 'f5'
	
	
	print('64*8, 64*32, 64*32')
	model = PoorEyesightModel.PoorEyesightSquareModel(square, 64*8, 64*32, 64*32)
	model.load_state_dict(torch.load(f'./model_{square}_648_6432_6432.pt'))
	model.to(device)
	eval(model,square,test_data_df)
	
	square = 'g4'
	
	print('64*8, 64*32, 64*32')
	model = PoorEyesightModel.PoorEyesightSquareModel(square, 64*8, 64*32, 64*32)
	model.load_state_dict(torch.load(f'./model_{square}_648_6432_6432.pt'))
	model.to(device)
	eval(model,square,test_data_df)
	
	print('64*8, 64*32, 64*32, 64')
	model = PoorEyesightModel.PoorEyesightSquareModel4(square, 64*8, 64*32, 64*32, 64)
	model.load_state_dict(torch.load(f'./model_{square}_648_6432_6432_64.pt'))
	model.to(device)
	eval(model,square,test_data_df)
	
	
	print('64*8, 64*32, 64*32, 64*2')
	model = PoorEyesightModel.PoorEyesightSquareModel4(square, 64*8, 64*32, 64*32, 64*2)
	model.load_state_dict(torch.load(f'./model_{square}_648_6432_6432_642.pt'))
	model.to(device)
	eval(model,square,test_data_df)
	
	
	print('64*8, 64*32, 64*32, 64*4')
	model = PoorEyesightModel.PoorEyesightSquareModel4(square, 64*8, 64*32, 64*32, 64*4)
	model.load_state_dict(torch.load(f'./model_{square}_648_6432_6432_644.pt'))
	model.to(device)
	eval(model,square,test_data_df)
	
	print('64*8, 64*32, 64*32, 64*8')
	model = PoorEyesightModel.PoorEyesightSquareModel4(square, 64*8, 64*32, 64*32, 64*8)
	model.load_state_dict(torch.load(f'./model_{square}_648_6432_6432_648.pt'))
	model.to(device)
	eval(model,square,test_data_df)
	
	print('64*8, 64*32, 64*32, 64*13')
	model = PoorEyesightModel.PoorEyesightSquareModel4(square, 64*8, 64*32, 64*32, 64*13)
	model.load_state_dict(torch.load(f'./model_{square}_648_6432_6432_6413.pt'))
	model.to(device)
	eval(model,square,test_data_df)
	
	
	"""
	for square in SquareData.squares:
		model = PoorEyesightModel.PoorEyesightSquareModel(square, 64*16, 64*64, 64*32)
		model.to(device)
		optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
		train(model,optimizer,square)
		torch.save(model.state_dict(), f'./models/model_{square}.pt')
		eval(model,square)
	"""
		

if __name__ == '__main__':
	main()