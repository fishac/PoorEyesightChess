import itertools

# Set up square data.
files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
ranks = ['1', '2', '3', '4', '5', '6', '7', '8']
squares = [''.join(s) for s in itertools.product(*[files,ranks])]

# Set up occupants data. 
# All of the pieces of each color, plus an empty disignation.
# First an last rank cannot have pawns.
occupants = ['wk', 'bk', 'wq', 'bq', 'wr', 'br', 'wn', 'bn', 'wb', 'bb', 'x', 'wp', 'bp']
first_last_rank_occupants = ['wk', 'bk', 'wq', 'bq', 'wr', 'br', 'wn', 'bn', 'wb', 'bb', 'x']
occupant_indices = {
	'wk': 0, 
	'bk': 1, 
	'wq': 2, 
	'bq': 3, 
	'wr': 4, 
	'br': 5, 
	'wn': 6, 
	'bn': 7, 
	'wb': 8, 
	'bb': 9, 
	'x': 10, 
	'wp': 11, 
	'bp': 12
}

occupants_pretty = ['white king', 'black king', 'white queen', 'black queen', 'white rook', 'black rook', 'white knight', 'black knight', 'white bishop', 'black bishop', 'empty', 'white pawn', 'black pawn']
first_last_rank_occupants_pretty = ['white king', 'black king', 'white queen', 'black queen', 'white rook', 'black rook', 'white knight', 'black knight', 'white bishop', 'black bishop', 'empty']

# Set up occupants color data.
# White, black, empty.
colors = ['w', 'b']
colors_backup = ['w', 'b', 'x']

def get_empty_board():
	return {square: 'x' for square in squares}
	
def square_first_last_rank(square):
	return ('1' in square or '8' in square)
	
# Set up dict of total output classes per square
square_total_occupants = {}
for square in squares:
	if square_first_last_rank(square):
		square_total_occupants[square] = 11
	else:
		square_total_occupants[square] = 13
		
# Set up list/dict of input features 
input_features = []
input_features_dict = {}
for square in squares:
	square_input_features = []
	for color in colors:
		feature = square + '_' + color
		input_features.append(feature)
		square_input_features.append(feature)
	input_features_dict[square] = square_input_features
		
# Set up list/dict of output features
output_features = []
output_features_dict = {}
for square in squares:
	square_output_features = []
	if square_first_last_rank(square):
		for occupant in first_last_rank_occupants:
			feature = square + '_' + occupant
			output_features.append(feature)
			square_output_features.append(feature)
	else:
		for occupant in occupants:
			feature = square + '_' + occupant
			output_features.append(feature)
			square_output_features.append(feature)
	output_features_dict[square] = square_output_features

# Set up dict of output features by square
output_features_by_square = {}
for square in squares:
	output_features_by_square[square] = []
	if square_first_last_rank(square):
		for occupant in first_last_rank_occupants:
			output_features_by_square[square].append(square + '_' + occupant)
	else:
		for occupant in occupants:
			output_features_by_square[square].append(square + '_' + occupant)
			