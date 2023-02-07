import SquareData

piece_fen_map = {
    'wp': 'P', 'bp': 'p',
    'wn': 'N', 'bn': 'n',
    'wb': 'B', 'bb': 'b',
    'wr': 'R', 'br': 'r',
    'wq': 'Q', 'bq': 'q',
    'wk': 'K', 'bk': 'k'
}

def get_fen(board_state):
    board_fen_list = []
    for rank in SquareData.ranks[::-1]:
        rank_fen_list = []
        empty_square_counter = 0
        for file in SquareData.files:
            square = file + rank
            if board_state[square] == 'x':
                empty_square_counter += 1
            else:
                if empty_square_counter > 0:
                    rank_fen_list.append(str(empty_square_counter))
                    empty_square_counter = 0
                rank_fen_list.append(piece_fen_map[board_state[square]])
        if empty_square_counter > 0:
            rank_fen_list.append(str(empty_square_counter))
        rank_fen = ''.join(rank_fen_list)
        board_fen_list.append(rank_fen)
    return '/'.join(board_fen_list)