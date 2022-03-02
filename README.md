# mctsGo
This project is written in C++ and CUDA, it contains:
1. CPU implementation of MCTS
2. CUDA implementation of MCTS
3. user interface for playing
4. module for communication between two players

The played game is Go with some rules changed for simplification.

Playing modules communicate with each other via communication module using UNIX named pipes.

### CPU MCTS
This is a simple implementation of Monte-Carlo Tree Search algorithm using single thread. 
Color, seconds spent per move and simulations count for each node can be changed.

### CUDA MCTS
Implementation of Monte-Carlo Tree Search. CUDA is used for parallelization of simulations performed.
The parallelization method implemented is leaf parallelziation but modified in order to use multiple CUDA blocks.
Color and seconds per move can be changed. It still can be improved in terms of performance.
