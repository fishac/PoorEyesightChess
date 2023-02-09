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

total_files_train = 55
total_files_test = 10
batch_size = 10000
total_file_entries = 100000

def get_file_output_dict():
	output_dict = {}
	for square in SquareData.squares:
		output_dict[square + '_actual'] = []
		output_dict[square + '_pred'] = []
	return output_dict

def main():
	models = {}
	print('Loading Models',flush=True)
	for square in SquareData.squares:
		model = PoorEyesightModel.PoorEyesightSquareModel(square, 64*8, 64*32, 64*32, 64*4)
		model.load_state_dict(torch.load(f'./models/model_{square}.pt'))
		model.to('cpu')
		models[square] = model
	print('Finished Loading Models',flush=True)
	print('Testing',flush=True)
	for i in range(total_files_test):
		print(f'Testing file {i}',flush=True)
		test_data_df = pd.read_csv(f'./processed_data/processed_data_{i+total_files_train}.csv')
		output_dict = get_file_output_dict()
		for square in SquareData.squares:
			#print(f'\tTesting square {square}',flush=True)
			n_test_batches = ceil(total_file_entries/batch_size)
			data_idx = 0
			
			for batch in range(n_test_batches):
				#print(f'\tTesting batch {batch}',flush=True)
				test_input = torch.FloatTensor(test_data_df[SquareData.input_features].values)[data_idx:min([(data_idx+batch_size),total_file_entries])]
				test_output = torch.FloatTensor(test_data_df[SquareData.output_features_dict[square]].values)[data_idx:min([(data_idx+batch_size),total_file_entries])]
				
				test_out_prob = models[square](test_input)
				test_out_pred = torch.argmax(test_out_prob,dim=1).tolist()
				test_out_actual = torch.argmax(test_output,dim=1).tolist()
				
				output_dict[square + '_actual'] += test_out_actual
				output_dict[square + '_pred'] += test_out_pred 
				
				data_idx += batch_size
			
		output_df = pd.DataFrame(output_dict)
		output_df.to_csv(f'./test_results/{i+total_files_train}.csv',index=False)
			
if __name__ == '__main__':
	main()