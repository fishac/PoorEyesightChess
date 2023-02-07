# Poor Eyesight Chess

## Overview 

Oh no, [Stockfish](https://en.wikipedia.org/wiki/Stockfish_(chess))  forgot its glasses at home! The state of the art chess move-suggestion engine can only recognize the **location** and **color** of pieces on the board and must guess the piece types based on previously-observed games before it thinks of a move to make. 
While Stockfish with perfect vision (i.e., being told the exact board state) is far better than any human chess grandmaster, how well can it perform when it has "poor eyesight" and might incorrectly guess which pieces are on the board, influencing its decisions?

The primary goal of this project is to build a deep-learning model which uses only chess piece **location** and **color** information to predict the exact board state, trained on a history of chess games played by real people. 

Even the most interesting models are mere curiosities without proper deployment, so I created a small chess program to play against this "Poor Eyesight Stockfish" locally on the command-line, instructions for which can be found in the following Section. Additionally, I created a [Lichess](https://www.lichess.org) "bot" account, [PoorEyesightBOT](https://lichess.org/?user=PoorEyesightBot#friend), where you can play against Poor Eyesight Stockfish online with no account, downloads, or setup required! 

See the full report on this project: [PoorEyesightChessReport.pdf][PoorEyesightChessReport.pdf].

## Playing Against Poor Eyesight Stockfish Locally

First, download this repository via git or a zipped archive download. 

Playing against Poor Eyesight Stockfish locally requires a command-line interface, a Python installation with version >= 3.7, an installation of the [PyTorch](https://pytorch.org/) Python library (with CUDA optional), and an installation of the [pandas](https://pandas.pydata.org/) python library. I recommend installing pandas with `pip install pandas`, but following the instructions for installing PyTorch from their website.

An installation of Stockfish is also required. It is simple enough to download and unzip from [their website](https://stockfishchess.org/download/). With Stockfish is installed, edit the file `stockfish_location.sh` to point to your local Stockfish executable on your computer.

When all of this setup is complete, run `source stockfish_location.sh` on the command-line from inside this repository. This enables the python code to read the location of Stockfish. Finally, play a game by running `python PlayPoorEyesightBot.py` .