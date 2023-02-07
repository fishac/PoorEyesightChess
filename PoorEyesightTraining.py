import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time
import SquareData
import PoorEyesightModel
from math import ceil
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_epochs = 20
batch_size = 1000
total_files_train = 55
total_files_test = 10
criterion = nn.CrossEntropyLoss()

def get_replication_multiplier(n):
    return ceil(int(40000/n))
	
def replicate_rows(df,square):
	df_square = df[SquareData.input_features + SquareData.output_features_dict[square]].copy()
	value_counts = {}
	if SquareData.square_first_last_rank(square):
		value_counts = {occupant: df_square[square + '_' + occupant].sum() for occupant in SquareData.first_last_rank_occupants}
	else:
		value_counts = {occupant: df_square[square + '_' + occupant].sum() for occupant in SquareData.occupants}
	replication_factors = {occupant: get_replication_multiplier(value_counts[occupant]) for occupant in value_counts.keys()}
	replicated_dfs = []
	for occupant in value_counts.keys():
		df_to_replicate = df_square[df_square[square + '_' + occupant]==1]
		replicated_df = pd.DataFrame(np.repeat(df_to_replicate.values,replication_factors[occupant],axis=0))
		replicated_df.columns = df_square.columns
		replicated_dfs.append(replicated_df)
	df_square_out = pd.concat([df_square,*replicated_dfs],axis=0)
	return df_square_out.sample(frac=1).reset_index(drop=True)
	
def train(model,optimizer,square):
	model.train()
	print(f'Training model for square: {square}',flush=True)
	start = time.time()
	for epoch in range(n_epochs):
		epoch_loss = 0
		total_entries = 0
		for file_idx in range(total_files_train):
			#print(f'Loading file ./processed_data/processed_data_{file_idx}.csv')
			df = pd.read_csv(f'./processed_data/processed_data_{file_idx}.csv')
			#print(f'Replicating rows')
			df = replicate_rows(df,square)
			#print(f'Done replicating rows, new total: {len(df)}')
			input_data = torch.FloatTensor(df[SquareData.input_features].values).to(device)
			output_data = torch.FloatTensor(df[SquareData.output_features_dict[square]].values).to(device)
			total_entries += len(df)
			
			data_idx = 0
			n_batches = ceil(len(df)/batch_size)
			for batch_idx in range(n_batches):
				input_data_batch = input_data[data_idx:(data_idx+batch_size)]
				output_data_batch = output_data[data_idx:(data_idx+batch_size)]
				
				optimizer.zero_grad()
				output = model(input_data_batch)
				loss = criterion(output,output_data_batch)
				loss.backward()
				optimizer.step()
				data_idx += batch_size
				epoch_loss += loss
				#print(f'\tBatch {batch_idx} loss: {loss:.4f}, total time: {(time.time()-start):.2f}')
			#print(f'\tRunning file {file_idx} total loss: {epoch_loss:.4f}, average loss: {(epoch_loss/total_entries):.4f}, total time: {(time.time()-start):.2f}')
		print(f'\tEpoch {epoch} loss: {epoch_loss:.4f}, average loss: {(epoch_loss/total_entries):.4f}, total time: {(time.time()-start):.2f}',flush=True)

def main():
	squares = SquareData.squares
	if len(sys.argv) > 1:
		file = sys.argv[1]
		squares = list(filter(lambda x: file in x, squares))
	print(squares)
	for square in squares:
		model = PoorEyesightModel.PoorEyesightSquareModel(square, 64*8, 64*32, 64*32, 64*4)
		model.to(device)
		optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
		train(model,optimizer,square)
		torch.save(model.state_dict(), f'./models/model_{square}.pt')
		

if __name__ == '__main__':
	main()