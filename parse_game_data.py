import chess
import chess.pgn
import pandas as pd
import itertools
import sys

# Set up square data.
files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
ranks = ['1', '2', '3', '4', '5', '6', '7', '8']
squares = [''.join(s) for s in itertools.product(*[files,ranks])]

# Set up occupants keys. 
# All of the pieces of each color, plus an empty disignation.
occupants = ['wr', 'wn', 'wb', 'wq', 'wk', 'wp', 'br', 'bn', 'bb', 'bq', 'bk', 'bp', 'x']

# Location of downloaded pgn database file.
database_filename = 'C:/Users/alexpc2red/Documents/lichessgames/lichess_db_standard_rated_2016-03.pgn'

# Define a conversion from library color notation to ours.
def parse_piece_color(piece):
    if piece.color is None:
        return None
    elif piece.color:
        return 'w'
    else:
        return 'b'
		
# Determine if board state has not been seen yet.
# Return -1 if new board state, otherwise return index of seen board
def index_of_board_state(board_states, board_state_new):
    try: 
        return board_states.index(board_state_new)
    except ValueError:
        return -1

# Generate board state dict from a given board object.
def parse_fen_position(board):
    fen_position = board.fen().split(' ')[0]
    return fen_position
	
# Generate a list of all board state dicts from the database file.
def parse_all_unique_fen_positions(max_games=-1):
	fen_positions = {}
	total_games = 0
	total_states = 0
	pgn = open(database_filename)
	while True:
		if total_games > max_games and max_games > 0:
			break
		game = chess.pgn.read_game(pgn)
		if game is None:
			break
		board = game.board()
		fen_position = parse_fen_position(board)
		total_states += 1
		if fen_position not in fen_positions.keys():
			fen_positions[fen_position] = 1
		else:
			fen_positions[fen_position] += 1
		for move in game.mainline_moves():
			board.push(move)
			fen_position = parse_fen_position(board)
			total_states += 1
			if fen_position not in fen_positions.keys():
				fen_positions[fen_position] = 1
			else:
				fen_positions[fen_position] += 1
		total_games += 1 
		
	fen_positions_list = fen_positions.keys()
	print(f'Total states observed: {total_states}')
	print(f'Total unique states: {len(fen_positions_list)}')
	print(f'Total nonunique states: {total_states-len(fen_positions_list)}')
	return fen_positions_list
	
# Define a conversion from library color notation to ours.
def parse_piece_color(piece):
	if piece.color is None:
		return None
	elif piece.color:
		return 'w'
	else:
		return 'b'
	
# Create new board state dict representing empty board.
def generate_empty_board_state():
	board_state = {}
	for square in squares:
		board_state[square] = 'x'
	return board_state
	
def parse_board_state(board):
	# Open database file.
	board_state = generate_empty_board_state()
	for square in squares:
		piece = board.piece_at(chess.parse_square(square))
		if piece is not None:
			color = parse_piece_color(piece)
			piece_char = piece.symbol().lower()
			occupant = color + piece_char
			board_state[square] = occupant
	return board_state
	
# Convert stored board state FENs to datasets we can use.
def convert_fens_to_df(fen_positions):
	board_states = []
	board = chess.Board()
	for fen_position in fen_positions:
		board.set_fen(fen_position)
		board_state = parse_board_state(board)
		board_states.append(board_state)
	board_states_df = pd.DataFrame(board_states)
	return board_states_df

	
def main():
	max_games = -1
	if len(sys.argv) > 1:
		max_games = int(sys.argv[1])
	fen_positions = parse_all_unique_fen_positions(max_games)
	board_states_df = convert_fens_to_df(fen_positions)
	board_states_df.to_csv('./board_states.csv',index=False)
	
	


if __name__ == '__main__':
	main()