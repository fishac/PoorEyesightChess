import chess
import chess.engine
import SquareData
import FEN
import PoorEyesightModel
import torch
from random import randint
import os

class PoorEyesightBot:
	def __init__(self):
		self.board = chess.Board()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.engine = chess.engine.SimpleEngine.popen_uci(os.environ['STOCKFISH_EXE'])
		self.load_models()
		self.memory = []
		self.MEMORY_LENGTH = 3
	
	def set_conversation(self, conversation):
		self.conversation = conversation
	
	def load_models(self):
		self.models = {}
		for square in SquareData.squares:
			model = PoorEyesightModel.PoorEyesightSquareModel(square, 64*8, 64*32, 64*32, 64*4)
			model.load_state_dict(torch.load(f'./models/model_{square}.pt'))
			model.to(self.device)
			self.models[square] = model
			
	def get_model_input(self,real_board):
		input = torch.zeros(1,128)
		i = 0
		for square in SquareData.squares:
			piece = real_board.piece_at(chess.parse_square(square))
			if piece is not None:
				if piece.color == chess.WHITE:
					input[0,i] = 1
				elif piece.color == chess.BLACK:
					input[0,i+1] = 1
			i += 2
		return input.to(self.device)
	
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
			board_state_probs[square] = square_pred[0]
		return board_state,board_state_probs
	
	def fix_board_state(self, real_board, board_state, board_state_probs):
		total_wk = self.total_kings_color(board_state,'wk') 
		total_bk = self.total_kings_color(board_state,'bk') 
		if total_wk == 0:
			# Place white king if no white kings on board
			highest_wk_prob_square = 'a1'
			highest_wk_prob = 0
			wk_prob = 0
			for square in SquareData.squares:
				wk_prob = board_state_probs[square][SquareData.occupant_indices['wk']]
				if wk_prob > highest_wk_prob:
					highest_wk_prob_square = square
					highest_wk_prob = wk_prob
			board_state[highest_wk_prob_square] = 'wk'
		
		if total_wk > 1:
			# Place second-most likely piece in place of least likely white kings if multiple white kings
			wk_squares = []
			wk_probs = []
			wk_highest_prob = 0
			wk_prob = 0
			for square in SquareData.squares:
				if board_state[square] == 'wk':
					wk_squares.append(square)
					wk_prob = board_state_probs[square][SquareData.occupant_indices['wk']].item()
					wk_probs.append(wk_prob)
					if wk_prob > wk_highest_prob:
						wk_highest_prob = wk_prob
						
			for i in range(total_wk):
				if wk_probs[i] < wk_highest_prob:
					sorted_probs = list(board_state_probs[wk_squares[i]])
					sorted_probs.sort()
					second_highest_prob  = sorted_probs[-2]
					second_highest_prob_index = board_state_probs[wk_squares[i]].tolist().index(second_highest_prob)
					board_state[wk_squares[i]] = SquareData.occupants[second_highest_prob_index]
					
		if total_bk == 0:
			# Place black king if no black kings on board
			highest_bk_prob_square = 'a1'
			highest_bk_prob = 0
			bk_prob = 0
			for square in SquareData.squares:
				bk_prob = board_state_probs[square][SquareData.occupant_indices['bk']]
				# Don't overwrite white king
				if bk_prob > highest_bk_prob and board_state[square] != 'wk':
					highest_bk_prob_square = square
					highest_bk_prob = bk_prob
			board_state[highest_bk_prob_square] = 'bk'
			
		# Place second-most likely piece in place of least likely black kings if multiple black kings
		if total_bk > 1:
			bk_squares = []
			bk_probs = []
			bk_highest_prob = 0
			bk_prob = 0
			# Find all squares with a black king and store their black king probabilities
			for square in SquareData.squares:
				if board_state[square] == 'bk':
					bk_squares.append(square)
					bk_prob = board_state_probs[square][SquareData.occupant_indices['bk']].item()
					bk_probs.append(bk_prob)
					if bk_prob > bk_highest_prob:
						bk_highest_prob = bk_prob
						
			# Set all black king squares except the most likely black king to the next-most likely pieces
			for i in range(total_bk):
				if bk_probs[i] < bk_highest_prob:
					sorted_probs = list(board_state_probs[bk_squares[i]])
					sorted_probs.sort()
					second_highest_prob  = sorted_probs[-2]
					second_highest_prob_index = board_state_probs[bk_squares[i]].tolist().index(second_highest_prob)
					# If the next most likely piece is a white king, place 3rd most likely piece.
					if second_highest_prob_index != SquareData.occupant_indices['wk']:
						board_state[bk_squares[i]] = SquareData.occupants[second_highest_prob_index]
					else:
						third_highest_prob  = sorted_probs[-3]
						third_highest_prob_index = board_state_probs[bk_squares[i]].tolist().index(third_highest_prob)
						board_state[bk_squares[i]] = SquareData.occupants[third_highest_prob_index]
						
		
		self.recall_moves(real_board,board_state)
				
	
	def recall_moves(self, real_board, board_state):
		for remembered_move in self.memory:
			remembered_square = remembered_move.split('_')[0]
			remembered_piece = remembered_move.split('_')
			my_color = remembered_piece[0]
			piece_at_square = real_board.piece_at(chess.parse_square(remembered_square))
			if piece_at_square:
				if (piece_at_square.color == chess.WHITE and my_color == 'w') or (piece_at_square.color == chess.BLACK and my_color == 'b'):
					board_state[remembered_square] = remembered_piece
	
	def get_fen_with_metadata(self, board_state, real_board):
		predicted_fen = FEN.get_fen(board_state)
		real_fen_with_metadata = real_board.fen()
		real_fen_split = real_fen_with_metadata.split(' ')
		real_fen_split[0] = predicted_fen
		predicted_fen_with_metadata = ' '.join(real_fen_split)
		return predicted_fen_with_metadata
	
	def get_random_move(self, real_board):
		random_move_idx = randint(0,real_board.legal_moves.count()-1)
		random_move = list(real_board.legal_moves)[random_move_idx]
		return random_move
		
	def remember_move(self, move, real_board):
		piece = real_board.piece_at(move.from_square)
		piece_color = 'w'
		piece_type = piece.piece_type
		if piece.color == chess.BLACK:
			piece_color = 'b'
		if move.promotion:
			piece_type = move.promotion 
		piece_type_char = chess.piece_symbol(piece_type).lower()
		remembered_move = chess.square_name(move.to_square) + '_' + piece_color + piece_type_char 
		self.memory.append(remembered_move)
		if len(self.memory) > self.MEMORY_LENGTH:
			self.memory = self.memory[1:]
		
	
	def make_move(self, real_board, move_time_limit=1, print_predicted_board=False, using_as_engine=False):
		predicted_board_state,board_state_probs = self.predict_board_state(real_board)
		self.fix_board_state(real_board,predicted_board_state,board_state_probs)
		predicted_fen = self.get_fen_with_metadata(predicted_board_state,real_board)
		self.board.set_fen(predicted_fen)
		if print_predicted_board:
			print('\nAI\'s predicted board:',flush=True)
			print(self.board,flush=True)
		
		predicted_fen_board_only = predicted_fen.split(' ')[0]
		#self.conversation.send_message("player", f'Before my move, I think the board looks like this (in FEN notation): {predicted_fen_board_only}')
		engine_result = False
		engine_crash = False
		try:
			engine_result = self.engine.play(self.board, chess.engine.Limit(time=1))
		except:	
			self.engine.close()
			self.engine = chess.engine.SimpleEngine.popen_uci(os.environ['STOCKFISH_EXE'])
			engine_crash = True
		
		if engine_result:
			suggested_move = engine_result.move
			print(real_board.legal_moves,flush=True)
			if suggested_move in real_board.legal_moves:
				self.remember_move(suggested_move,real_board)
				return (suggested_move,False)
			elif len(list(real_board.legal_moves)) > 0:
				random_move = self.get_random_move(real_board)
				self.remember_move(random_move,real_board)
				return (random_move,True)
			else:
				return (None,True)
		elif engine_crash:
			random_move = self.get_random_move(real_board)
			self.remember_move(random_move,real_board)
			return (random_move,True)
		else:
			return (None,False)


	