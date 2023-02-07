import pandas as pd
import itertools
import time
import sys

# Set up square data.
files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
ranks = ['1', '2', '3', '4', '5', '6', '7', '8']
squares = [''.join(s) for s in itertools.product(*[files,ranks])]

# Set up occupants data. 
# All of the pieces of each color, plus an empty disignation.
# First an last rank cannot have pawns.
occupants = ['wr', 'wn', 'wb', 'wq', 'wk', 'wp', 'br', 'bn', 'bb', 'bq', 'bk', 'bp', 'x']
first_last_rank_occupants = ['wr', 'wn', 'wb', 'wq', 'wk', 'br', 'bn', 'bb', 'bq', 'bk', 'x']

# Set up occupants color data.
# White, black, empty.
colors = ['w', 'b', 'x']

def is_color(color):
	def is_color_elem(elem):
		return int(elem[0]==color)
	return is_color_elem
	
def is_occupant(occupant):
	def is_occupant_elem(elem):
		return int(elem == occupant)
	return is_occupant_elem
	
def process_data(parsed_data):
	processed_data = {}
	for square in squares:
		rank = square[1]
		for color in colors:
			processed_data[square + '_' + color] = (parsed_data[square].str[0] == color).astype(int)
		
		if rank == '1' or rank == '8':
			for occupant in first_last_rank_occupants:
				processed_data[square + '_' + occupant] = (parsed_data[square] == occupant).astype(int)
		else:
			for occupant in occupants:
				processed_data[square + '_' + occupant] = (parsed_data[square] == occupant).astype(int)
	return pd.DataFrame(processed_data)

def main():
	start = time.time()
	if len(sys.argv) >= 2:
		if sys.argv[1] == 'all':
			parsed_data = pd.read_csv('./board_states_trimmed_shuffled.csv')
			processed_data = process_data(parsed_data)
			processed_data.to_csv('./processed_data/processed_data_0.csv',index=False)
			checkpoint_duration = time.time() - start
			print(f'Finished processing chunk: 0. Total duration: {checkpoint_duration} seconds.',flush=True)
		else:
			entries_per_file = int(sys.argv[1])
			chunk_index = 0
			for parsed_data in pd.read_csv('./board_states_trimmed_shuffled.csv',chunksize=entries_per_file):
				processed_data = process_data(parsed_data)
				processed_data.to_csv(f'./processed_data/processed_data_{chunk_index}.csv',index=False)
				checkpoint_duration = time.time() - start
				print(f'Finished processing chunk: {chunk_index}. Total duration: {checkpoint_duration} seconds.',flush=True)
				chunk_index += 1
	else:
		print('Requires one command line argument: number of rows per output file. Or the word \'all\' for one output file.')

if __name__ == '__main__':
	main()