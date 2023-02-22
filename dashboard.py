import streamlit as st 
import chess
import chess.svg
import SquareData
import FEN
import PoorEyesightModel
import torch
import PoorEyesightPredictor
from cairosvg import svg2png

initial_board_state_data_indices = {}
for square in SquareData.squares:
	if square[1] == '1' or square[1] == '8':
		initial_board_state_data_indices[square] = 10
	else:
		initial_board_state_data_indices[square] = 12
for file in SquareData.files:
	initial_board_state_data_indices[file + '2'] = 0
	initial_board_state_data_indices[file + '7'] = 1
initial_board_state_data_indices['a1'] = 4
initial_board_state_data_indices['b1'] = 0
initial_board_state_data_indices['c1'] = 2
initial_board_state_data_indices['d1'] = 6
initial_board_state_data_indices['e1'] = 8
initial_board_state_data_indices['f1'] = 2
initial_board_state_data_indices['g1'] = 0
initial_board_state_data_indices['h1'] = 4
initial_board_state_data_indices['a8'] = 5
initial_board_state_data_indices['b8'] = 1
initial_board_state_data_indices['c8'] = 3
initial_board_state_data_indices['d8'] = 7
initial_board_state_data_indices['e8'] = 9
initial_board_state_data_indices['f8'] = 3
initial_board_state_data_indices['g8'] = 1
initial_board_state_data_indices['h8'] = 5

predictor = PoorEyesightPredictor.PoorEyesightPredictor()

real_board = chess.Board()
chess.svg.board(board=real_board,size=400)
model_board = chess.Board()

occupants_dropdown = ['White Pawn', 'Black Pawn', 'White Knight', 'Black Knight', 'White Bishop', 'Black Bishop', 'White Rook', 'Black Rook','White Queen', 'Black Queen', 'White King', 'Black King', 'Empty']
occupants_dropdown_firstlast = ['White Knight', 'Black Knight', 'White Bishop', 'Black Bishop', 'White Rook', 'Black Rook','White Queen', 'Black Queen', 'White King', 'Black King', 'Empty']

st.title('Poor Eyesight Model')

board_state_data = {}

def render_file(file):
	for rank in SquareData.ranks:
		if rank == '1' or rank == '8':
			board_state_data[file + rank] = st.selectbox(file+rank,occupants_dropdown_firstlast,index=initial_board_state_data_indices[file + rank])
		else:
			board_state_data[file + rank] = st.selectbox(file+rank,occupants_dropdown,index=initial_board_state_data_indices[file + rank])
	
def convert_board_state(board_state_data):
	board_state = board_state_data.copy()
	for square in SquareData.squares:
		if board_state_data[square] == 'Empty':
			board_state[square] = 'x'
		else:
			color = board_state_data[square].split(' ')[0][0].lower()
			piece = board_state_data[square].split(' ')[1][0].lower()
			if 'Knight' in board_state_data[square]:
				piece = 'n'
			board_state[square] = color + piece
	return board_state

ae_col, bf_col, cg_col, dh_col = st.columns(4)

with ae_col:
	render_file('a')
	render_file('e')

with bf_col:
	render_file('b')
	render_file('f')
	
with cg_col:
	render_file('c')
	render_file('g')
	
with dh_col:
	render_file('d')
	render_file('h')
	
def set_model_board(real_board):
	model_board_state,model_board_state_probs = predictor.predict_board_state(real_board)
	model_board_fen = FEN.get_fen(model_board_state)
	model_board.set_fen(model_board_fen)
	
real_board = chess.Board(FEN.get_fen(convert_board_state(board_state_data)))
set_model_board(real_board)

wrong_squares = []
for square in SquareData.squares:
	chess_square = chess.parse_square(square)
	real_board_piece = real_board.piece_at(chess_square)
	model_board_piece = model_board.piece_at(chess_square)
	if not real_board_piece and model_board_piece:
		wrong_squares.append(chess_square)
		continue
	if real_board_piece and not model_board_piece:
		wrong_squares.append(chess_square)
		continue
	if real_board_piece and model_board_piece:
		if real_board_piece.symbol() != model_board_piece.symbol():
			wrong_squares.append(chess_square)
			continue

# python-chess libary has issues with displaying from SVG string, tends to display empty board
# Save to png and read as workaround
svg2png(bytestring=chess.svg.board(board=real_board,size=400),write_to='./real_board.png')
svg2png(bytestring=chess.svg.board(board=model_board,size=400,fill=dict.fromkeys(chess.SquareSet(wrong_squares),'#cc0000cc')),write_to='./model_board.png')

real_col,model_col = st.columns(2)

with real_col:
	st.image('./real_board.png',caption='Exact Board',use_column_width=True)
	
with model_col:
	st.image('./model_board.png',caption='Predicted Board',use_column_width=True)
