import pandas as pd
import itertools

# Set up square data.
files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
ranks = ['1', '2', '3', '4', '5', '6', '7', '8']
squares = [''.join(s) for s in itertools.product(*[files,ranks])]

# Set up occupants data. 
# All of the pieces of each color, plus an empty disignation.
occupants = ['wr', 'wn', 'wb', 'wq', 'wk', 'wp', 'br', 'bn', 'bb', 'bq', 'bk', 'bp', 'x']

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
	processed_data = pd.DataFrame()
	for square in squares:
		for color in colors:
			parsed_column = parsed_data[square].apply(is_color(color))
			parsed_column.name = square + '_' + color
			processed_data = pd.concat([processed_data,parsed_column])
		for occupant in occupants:
			# Pawns cannot exist on first or eighth ranks, no need to check.
			if (square[1] == '1' or square[0] == '8') and (occupant == 'wp' or occupant == 'bp'):
				continue
			else:
				parsed_column = parsed_data[square].apply(is_occupant(occupant))
				parsed_column.name = square + '_' + occupant
				processed_data = pd.concat([processed_data,parsed_column])
	return processed_data
	
def main():
	parsed_data = pd.read_csv('board_states.csv')
	processed_data = process_data(parsed_data)
	print(f'Columns: {processed_data.columns}')
	print(f'Total columns: {len(processed_data.columns)}')
	print('First 2 rows:')
	print(processed_data.head(2))
	
if __name__ == '__main__':
	main()