import chess
import SquareData
import FEN
import PoorEyesightModel
import torch

class PoorEyesightPredictor:
	def __init__(self):
		self.board = chess.Board()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.load_models()
	
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

	
	def get_fen_with_metadata(self, board_state, real_board):
		predicted_fen = FEN.get_fen(board_state)
		real_fen_with_metadata = real_board.fen()
		real_fen_split = real_fen_with_metadata.split(' ')
		real_fen_split[0] = predicted_fen
		predicted_fen_with_metadata = ' '.join(real_fen_split)
		return predicted_fen_with_metadata


	