# Poor Eyesight Chess

Oh no, [Stockfish](https://en.wikipedia.org/wiki/Stockfish_(chess))  forgot its glasses at home! The state of the art chess move-suggestion engine can only recognize the location and color of pieces on the board and must guess the piece types based on previously-observed games before it thinks of a move to make. 
While Stockfish with perfect vision (i.e., being told the exact board state) is far better than any human chess grandmaster, how well can it perform when it has "poor eyesight" and might incorrectly guess which pieces are on the board, influencing its decisions?

The primary goal of this project is to build a deep-learning model which uses only chess piece location and color information to predict the exact board state, trained on a history of chess games played by real people. 

Even the most interesting models are mere curiosities without proper deployment, so I created a small chess program to play against this "Poor Eyesight Stockfish" locally on the command-line, instructions for which can be found in the following Section. Additionally, I created a [Lichess](https://www.lichess.org) "bot" account, [PoorEyesightStockfish REPLACELINK](https://www.lichess.org), where you can play against Poor Eyesight Stockfish online with no account, downloads, or setup required! 