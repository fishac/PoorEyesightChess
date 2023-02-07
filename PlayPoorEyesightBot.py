import PoorEyesightBot
import chess
import SquareData


def main():
	board = chess.Board()
	bot = PoorEyesightBot.PoorEyesightBot()
	print('\nINTRO\nYou have challenged a chess AI with poor eyesight! The AI (Stockfish, with the black pieces) can only tell the color and location of pieces on board, and guesses the board state when it makes a move! If Stockfish suggests an illegal move, a legal move will be made at random for it and it can try again next turn.\n\nMOVES\nTo play a move, type the origin square for your piece and the destination square for the piece. For example, to move a pawn from e2 to e4, type: e2e4. \nTo move a pawn and promote, enter the move along with the piece to promote to. For example, to move a pawn from e7 to e8 and promote to a queen, type: e7e8q.\nTo castle kingside, type: e1g1. \nTo castle queenside, type: e1c1. \nTo resign, type: resign.\n\nBOARD\nThe columns (files) from left to right are labelled a through h, and the rows (ranks) from bottom to top are labelled 1 through 8.Capital letters (PNBRQK) represent white\'s pieces, lowercase letters (represent black\s pieces, dots (.) represents an empty square.\n\n')
	while True:
		print(board)
		print('\n')
		valid_move = False
		resigned = False
		move_from_uci = chess.Move.from_uci('e2e3')
		while not valid_move:
			move = input('Enter a move: ')
			if len(move) == 4 and move[0] in SquareData.files and move[1] in SquareData.ranks and move[2] in SquareData.files and move[3] in SquareData.ranks:
				move_from_uci = chess.Move.from_uci(move)
				valid_move = move_from_uci in board.legal_moves
				if not valid_move:
					print('Illegal move.')
			elif move == 'resign':
				print('You resign.')
				resigned = True
				valid_move = True
			else:
				print('Invalid input.')
		
		if resigned:
			break
			
		board.push(move_from_uci)
		
		if board.is_checkmate():
			print('Checkmate. You win.')
			break
			
		move,is_random = bot.make_move(board,print_predicted_board=True)
		board.push(move)
		if is_random:
			print('\nAI made a random move.')
		else:
			print('\nAI did not make a random move.')
			
		if board.is_checkmate():
			print('Checkmate. You lose.')
			break


if __name__ == '__main__':
	main()