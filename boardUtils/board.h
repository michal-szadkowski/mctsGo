#ifndef __BOARD
#define __BOARD

#include <cstdio>
#include <iostream>
#include <cstring>
#define BOARDSIZE 9
#define BOARDBYTES 21
#define MAXMOVES 300
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

	void Start(Board board);

	uint8_t GetPlayer(Board board);
	uint8_t SwitchPlayer(Board board);

	void PrintBoard(Board board);

	void Copy(Board src, Board dst);

	int MakeMove(Board board, uint col, uint row, uint8_t color, uint8_t isPass = 0);
	int MakeMove(Board board, Move move);
	int IsLegal(Board board, uint col, uint row, uint8_t color);
	uint GetMoves(Board board, Move *moveArr, uint cArr);

	void DeleteChainIfNoLiberties(Board b, uint col, uint row);
	void DeleteChain(Board board, uint col, uint row);
	void DeleteChainRec(Board board, uint col, uint row, uint8_t color);
	int CheckLibertiesChain(Board board, uint col, uint row);
	int CheckStoneLiberties(Board board, uint col, uint row);
	int CheckPositionLiberties(Board board, uint col, uint row);
	int CheckLibertiesRec(Board visited, Board board, uint col, uint row, uint8_t color);
	int CheckIfEye(Board board, uint col, uint row, uint8_t color);

	// int CheckWin(Board board, Move *prevMoves);
	int GetScores(Board board, int &black, int &white);

	void SetPosition(Board board, uint position, uint8_t val);
	void SetPosition(Board board, uint col, uint row, uint8_t val);
	uint8_t GetPosition(Board board, uint position);
	uint8_t GetPosition(Board board, uint col, uint row);
	uint8_t OpositeCol(uint8_t col);

	Move ComposeMove(uint col, uint row, uint8_t color, uint8_t isLastMove, uint8_t isPass);
	Move SetLastMoveFlag(Move mv, uint8_t lastMoveFlag);
	void DecomposeMove(Move move, uint &col, uint &row, uint8_t &color, uint8_t &isLastMove, uint8_t &isPass);
	void PrintMove(Move mv);
	uint xy2c(uint col, uint row);
	void c2xy(uint pos, uint &col, uint &row);
}
#endif