import chess
import chess.engine
import SquareData
import FEN
import PoorEyesightModel
import torch
from random import randint

class PoorEyesightBot:
	def __init__(self):
		self.board = chess.Board()
		self.engine = chess.engine.SimpleEngine.popen_uci("C:/Users/alexpc2red/Documents/stockfish_15.1_win_x64_avx2/stockfish-windows-2022-x86-64-avx2.exe")
		self.load_models(model_filename,params_filename)
			
	def load_models(self, model_filename, params_filename):
		self.models = {}
		for square in SquareData.squares:
			model = PoorEyesightModel.PoorEyesightSquareModel4(square, 64*8, 64*32, 64*32, 64*4)
			model.load_state_dict(torch.load(f'./models/model_{square}.pt'))
			model.to(device)
			models[square] = model
			
	def get_model_input(self,real_board):
		input = torch.zeros(1,128)
		i = 0
		for square in SquareData.squares:
			if square[0] == 'w':
				input[i] = 1
			elif square[0] == 'b':
				input[i+1] = 1
			i += 2
		return input.to(device)
	
	def total_kings_color(self, board_state, king_key):
		total_king_count = 0
		for square in SquareData.squares:
			total_king_count += int(board_state[square] == king_key)
		return total_king_count

	def predict_board_state(self, real_board):
		board_state = SquareData.get_empty_board()
		board_state_probs = SquareData.get_empty_board()
		model_input = self.get_model_input(real_board)
		for square in SquareData.squares:
			square_pred = self.models[square](model_input)
			max_idx = torch.argmax(square_pred)
			predicted_piece = SquareData.occupants[max_idx]
			board_state[square] = predicted_piece
			board_state_probs[square] = square_pred
		return board_state,board_state_probs
	
	def fix_board_state(self, board_state, board_state_probs):
		total_wk = self.total_kings_color(board_state,'wk') 
		total_bk = self.total_kings_color(board_state,'bk') 
		
		# Place white king if no white kings on board
		if total_wk == 0:
			highest_wk_prob_square = 'a1'
			highest_wk_prob = 0
			wk_prob = 0
			for square in SquareData.squares:
				wk_prob = board_state_probs[square][SquareData.occupant_indices['wk']]
				if wk_prob > highest_wk_prob:
					highest_wk_prob_square = square
					highest_wk_prob = wk_prob
			board_state[highest_wk_prob_square] = 'wk'
		
		# Place second-most likely piece in place of least likely white kings if multiple white kings
		if total_wk > 1:
			wk_squares = []
			wk_probs = []
			wk_highest_prob = 0
			wk_prob = 0
			for square in SquareData.squares:
				if board_state[square] == 'wk':
					wk_squares.append(square)
					wk_prob = board_state_probs[square][SquareData.occupant_indices['wk']]
					wk_probs.append(wk_prob)
					if wk_prob > wk_highest_prob:
						wk_highest_prob = wk_prob
						
			for i in range(total_wk):
				if wk_probs[i] < wk_highest_prob:
					sorted_probs = list(board_state_probs[square]).sort()
					second_highest_prob  = sorted_probs[-2]
					second_highest_prob_index = board_state_probs[square].index(second_highest_prob)
					board_state[square] = SquareData.occupants[second_highest_prob_index]
					
		# Place black king if no black kings on board
		if total_bk == 0:
			highest_bk_prob_square = 'a1'
			highest_bk_prob = 0
			bk_prob = 0
			for square in SquareData.squares:
				bk_prob = board_state_probs[square][SquareData.occupant_indices['bk']]
				# Don't overwrite white king
				if bk_prob > highest_bk_prob and board_state[square] != 'wk':
					highest_bk_prob_square = square
					highest_bk_prob = wk_prob
			board_state[highest_bk_prob_square] = 'bk'
			
		# Place second-most likely piece in place of least likely black kings if multiple black kings
		if total_bk > 1:
			bk_squares = []
			bk_probs = []
			bk_highest_prob = 0
			bk_prob = 0
			for square in SquareData.squares:
				if board_state[square] == 'bk':
					bk_squares.append(square)
					bk_prob = board_state_probs[square][SquareData.occupant_indices['bk']]
					bk_probs.append(wk_prob)
					if bk_prob > bk_highest_prob:
						bk_highest_prob = bk_prob
						
			for i in range(total_bk):
				if wk_probs[i] < bk_highest_prob:
					sorted_probs = list(board_state_probs[square]).sort()
					second_highest_prob  = sorted_probs[-2]
					second_highest_prob_index = board_state_probs[square].index(second_highest_prob)
					# Don't place an extra white king on board
					if second_highest_prob_index != SquareData.occupant_indices['wk']:
						board_state[square] = SquareData.occupants[second_highest_prob_index]
					else:
						third_highest_prob  = sorted_probs[-3]
						third_highest_prob_index = board_state_probs[square].index(third_highest_prob)
						board_state[square] = SquareData.occupants[third_highest_prob_index]
				
	def get_fen_with_metadata(self, board_state, real_board):
		predicted_fen = FEN.get_fen(board_state)
		real_fen_with_metadata = real_board.fen()
		real_fen_split = real_fen_with_metadata.split(' ')
		real_fen_split[0] = predicted_fen
		predicted_fen_with_metadata = ' '.join(real_fen_split)
		return predicted_fen_with_metadata
	
	def make_move(self, real_board, move_time_limit=1, print_predicted_board=False):
		predicted_board_state,board_state_probs = self.predict_board_state(real_board)
		fixed_board_state = self.fix_board_state(predicted_board_state,board_state_probs)
		predicted_fen = self.get_fen_with_metadata(fixed_board_state,real_board)
		self.board.set_fen(predicted_fen)
		if print_predicted_board:
			print('\nAI\'s predicted board:')
			print(self.board)
		suggested_move = engine.play(self.board, chess.engine.Limit(time=1))
		if suggested_move in real_board.legal_moves:
			return (suggested_move,False)
		else:
			random_move_idx = randint(0,real_board.legal_moves.count()-1)
			random_move = list(real_board.legal_moves)[random_move_idx]
			return (random_move,True)
		
	def __del__(self):
		self.engine.quit()

	