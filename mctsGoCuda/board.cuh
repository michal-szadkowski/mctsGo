#include <cuda_runtime.h>

#ifndef __BOARD
#define __BOARD

#include <cstdio>
#include <iostream>
#include <cstring>
#define BOARDSIZE 9
#define BOARDBYTES 21
#define MAXMOVES 140
#define POSBITS 7 // log_2 BOARDSZIE*BOARDSIZE
#define BLACK 0
#define WHITE 1
#define KOMI 1

#define ERR(source) (perror(source),                                 \
					 fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), \
					 exit(EXIT_FAILURE))

namespace brd
{
	typedef unsigned char Board[BOARDBYTES];
	typedef uint16_t Move;

	/*Format info
	Position of move on first POSBITS bits
	Last bits store: pass flag, color of player and flag for last move in sequence
	*/

	__device__ __host__ void Start(Board board);

	__device__ __host__ uint8_t GetPlayer(Board board);
	__device__ __host__ uint8_t SwitchPlayer(Board board);

	void PrintBoard(Board board);

	__device__ __host__ void Copy(Board src, Board dst);

	__device__ __host__ __host__ int MakeMove(Board board, uint col, uint row, uint8_t color, uint8_t isPass = 0);
	__device__ __host__ int MakeMove(Board board, Move move);
	__device__ __host__ int IsLegal(Board board, uint col, uint row, uint8_t color);
	__device__ __host__ uint GetMoves(Board board, Move *moveArr, uint cArr);

	__device__ __host__ void DeleteChainIfNoLiberties(Board b, uint col, uint row);
	__device__ __host__ void DeleteChain(Board board, uint col, uint row);
	__device__ __host__ void DeleteChainRec(Board board, uint col, uint row, uint8_t color);
	__device__ __host__ int CheckLibertiesChain(Board board, uint col, uint row);
	__device__ __host__ int CheckStoneLiberties(Board board, uint col, uint row);
	__device__ __host__ int CheckPositionLiberties(Board board, uint col, uint row);
	__device__ __host__ int CheckLibertiesRec(Board visited, Board board, uint col, uint row, uint8_t color);
	__device__ __host__ int CheckIfEye(Board board, uint col, uint row, uint8_t color);

	// int CheckWin(Board board, Move *prevMoves);
	__device__ __host__ int GetScores(Board board, int &black, int &white);

	__device__ __host__ void SetPosition(Board board, uint position, uint8_t val);
	__device__ __host__ void SetPosition(Board board, uint col, uint row, uint8_t val);
	__device__ __host__ uint8_t GetPosition(Board board, uint position);
	__device__ __host__ uint8_t GetPosition(Board board, uint col, uint row);
	__device__ __host__ uint8_t OpositeCol(uint8_t col);

	__device__ __host__ Move ComposeMove(uint col, uint row, uint8_t color, uint8_t isLastMove, uint8_t isPass);
	__device__ __host__ Move SetLastMoveFlag(Move mv, uint8_t lastMoveFlag);
	__device__ __host__ void DecomposeMove(Move move, uint &col, uint &row, uint8_t &color, uint8_t &isLastMove, uint8_t &isPass);
	void PrintMove(Move mv);
	__device__ __host__ uint xy2c(uint col, uint row);
	__device__ __host__ void c2xy(uint pos, uint &col, uint &row);
}
#endif