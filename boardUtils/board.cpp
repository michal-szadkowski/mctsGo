#include <cstdio>
#include <iostream>
#include <cstring>
#include "board.h"

namespace brd
{
	void Start(Board board)
	{
		if (!board)
			ERR("Start");
		std::memset(board, 0, BOARDBYTES);
	}
	// Returns 0 if black to move, 1 otherwise
	uint8_t GetPlayer(Board board)
	{
		uint8_t player = (board[BOARDBYTES - 1] & 0b1);
		return player;
	}
	// Switches player to move
	uint8_t SwitchPlayer(Board board)
	{
		uint8_t player = board[BOARDBYTES - 1] & 1;
		player = ~player & 1;
		board[BOARDBYTES - 1] &= ~1;
		board[BOARDBYTES - 1] |= player;
		return player;
	}

	void PrintBoard(Board board)
	{
		std::cout << " ";
		for (int i = 0; i < BOARDSIZE; i++)
			std::cout << i;
		std::cout << "\n";
		for (int i = 0; i < BOARDSIZE; i++)
		{
			std::cout << (char)('a' + i);
			for (int j = 0; j < BOARDSIZE; j++)
			{
				uint8_t p = GetPosition(board, j, i);
				if (p == 2)
					std::cout << "\033[1;31m■\033[0m";
				else if (p == 3)
					std::cout << "\033[1;32m■\033[0m";
				else
					std::cout
						<< ".";
			}
			std::cout << "\n";
		}
		if (GetPlayer(board) == 0)
			std::cout << "\033[1;31m■\033[0m";
		else
			std::cout << "\033[1;32m■\033[0m";
		std::cout << "\n";
	}

	void Copy(Board src, Board dst)
	{
		if (!src || !dst)
			ERR("Copy");
		std::memcpy(dst, src, BOARDBYTES);
	}

	uint GetMoves(Board board, Move *moveArr, uint cArr)
	{
		uint8_t color = GetPlayer(board);
		uint addCount = 0;
		if (addCount < cArr)
		{
			moveArr[addCount] = ComposeMove(0, 0, color, 0, 1);
			addCount++;
		}
		else
			return 0;
		for (int i = 0; i < BOARDSIZE; i++)
		{
			for (int j = 0; j < BOARDSIZE; j++)
			{
				if (IsLegal(board, i, j, color) && addCount < cArr)
				{
					moveArr[addCount] = ComposeMove(i, j, color, 0, 0);
					addCount++;
				}
			}
		}
		SetLastMoveFlag(moveArr[addCount - 1], 1);
		return addCount;
	}

	// Makes move if legal
	int MakeMove(Board board, uint col, uint row, uint8_t color, uint8_t isPass)
	{
		if (isPass)
		{
			SwitchPlayer(board);
			return 0;
		}
		color += 2;
		if (IsLegal(board, col, row, color) == 1)
		{
			SetPosition(board, col, row, color);

			SwitchPlayer(board);

			if (col > 0)
				DeleteChainIfNoLiberties(board, col - 1, row);
			if (col < BOARDSIZE - 1)
				DeleteChainIfNoLiberties(board, col + 1, row);
			if (row > 0)
				DeleteChainIfNoLiberties(board, col, row - 1);
			if (row < BOARDSIZE - 1)
				DeleteChainIfNoLiberties(board, col, row + 1);
			DeleteChainIfNoLiberties(board, col, row);
			return 0;
		}
		return 1;
	}
	int MakeMove(Board board, Move move)
	{
		uint col, row;
		uint8_t color, last, pass;
		DecomposeMove(move, col, row, color, last, pass);
		return MakeMove(board, col, row, color, pass);
	}

	// Returns 1 if move is legal, 0 otherwise
	int IsLegal(Board board, uint col, uint row, uint8_t color)
	{
		if (col >= BOARDSIZE || row >= BOARDSIZE)
			return 0;
		if (color < 2)
			color += 2;
		if (color - 2 != GetPlayer(board))
			return 0;
		if (CheckIfEye(board, col, row, color) == 1)
			return 0;
		uint8_t bPos = GetPosition(board, col, row);
		if (bPos < 2)
			return 1;
		else
			return 0;
	}

	void DeleteChainIfNoLiberties(Board b, uint col, uint row)
	{
		uint chainLib = CheckLibertiesChain(b, col, row);
		if (chainLib == 0)
			DeleteChain(b, col, row);
	}
	// Deletes chain if position is not empty
	void DeleteChain(Board b, uint col, uint row)
	{
		uint8_t color = GetPosition(b, col, row);
		if (color < 2)
			return;
		DeleteChainRec(b, col, row, color);
	}
	void DeleteChainRec(Board b, uint col, uint row, uint8_t color)
	{
		uint8_t currentColor = GetPosition(b, col, row);
		if (currentColor < 2)
			return;
		if (currentColor != color)
			return;

		SetPosition(b, col, row, 0);
		if (col > 0)
			DeleteChainRec(b, col - 1, row, color);
		if (col < BOARDSIZE - 1)
			DeleteChainRec(b, col + 1, row, color);
		if (row > 0)
			DeleteChainRec(b, col, row - 1, color);
		if (row < BOARDSIZE - 1)
			DeleteChainRec(b, col, row + 1, color);
	}
	// Returns 0 it chain has no liberties
	int CheckLibertiesChain(Board board, uint col, uint row)
	{
		Board b;
		Start(b);
		uint16_t color = GetPosition(board, col, row);
		if (color < 2)
			return 1;
		int lib = CheckLibertiesRec(b, board, col, row, color);
		return lib;
	}
	int CheckLibertiesRec(Board visited, Board board, uint col, uint row, uint8_t color)
	{
		if (GetPosition(visited, col, row) == 2)
			return 0;
		SetPosition(visited, col, row, 2);
		if (GetPosition(board, col, row) == color)
		{

			int lib = CheckStoneLiberties(board, col, row);
			if (col > 0)
				lib += CheckLibertiesRec(visited, board, col - 1, row, color);
			if (col < BOARDSIZE - 1)
				lib += CheckLibertiesRec(visited, board, col + 1, row, color);
			if (row > 0)
				lib += CheckLibertiesRec(visited, board, col, row - 1, color);
			if (row < BOARDSIZE - 1)
				lib += CheckLibertiesRec(visited, board, col, row + 1, color);

			return lib;
		}
		return 0;
	}
	// Returns liberties of a stone, 0 if there is no stone or stone has no liberties
	int CheckStoneLiberties(Board board, uint col, uint row)
	{
		int c = GetPosition(board, col, row);
		if (c < 2)
			return 0;

		return CheckPositionLiberties(board, col, row);
	}

	int CheckPositionLiberties(Board board, uint col, uint row)
	{
		int lib = 0;
		uint8_t up, left, down, right;
		if (row > 0)
		{
			up = GetPosition(board, col, row - 1);
			if (up < 2)
				lib += 1;
		}
		else
		{
			lib += 1;
		}
		if (row < BOARDSIZE - 1)
		{
			down = GetPosition(board, col, row + 1);
			if (down < 2)
				lib += 1;
		}
		else
		{
			lib += 1;
		}
		if (col > 0)
		{
			left = GetPosition(board, col - 1, row);
			if (left < 2)
				lib += 1;
		}
		else
		{
			lib += 1;
		}
		if (col < BOARDSIZE - 1)
		{
			right = GetPosition(board, col + 1, row);
			if (right < 2)
				lib += 1;
		}
		else
		{
			lib += 1;
		}
		return lib;
	}

	// Retruns 1 if empty position is an Eye, 0 if its not empty or not an Eye
	int CheckIfEye(Board board, uint col, uint row, uint8_t color)
	{
		uint8_t c = GetPosition(board, col, row);
		if (c >= 2)
			return 0;
		uint8_t oposite = OpositeCol(color);

		int lib = 0;
		uint8_t up, left, down, right;
		if (row > 0)
		{
			up = GetPosition(board, col, row - 1);
			if (up != oposite)
				lib += 1;
		}
		if (row < BOARDSIZE - 1)
		{
			down = GetPosition(board, col, row + 1);
			if (down != oposite)
				lib += 1;
		}
		if (col > 0)
		{
			left = GetPosition(board, col - 1, row);
			if (left != oposite)
				lib += 1;
		}
		if (col < BOARDSIZE - 1)
		{
			right = GetPosition(board, col + 1, row);
			if (right != oposite)
				lib += 1;
		}
		if (lib == 0)
			return 1;
		return 0;
	}

	int CheckWin(Board board)
	{
		return 0;
	}

	int GetScores(Board board, int &black, int &white)
	{
		black = 0;
		white = 0;
		for (int i = 0; i < BOARDSIZE; i++)
		{
			for (int j = 0; j < BOARDSIZE; j++)
			{
				uint8_t pos = GetPosition(board, i, j);
				if (pos == 2)
				{
					black += 1;
				}
				if (pos == 3)
				{
					white += 1;
				}
			}
		}
		return BOARDSIZE * BOARDSIZE - (black + white);
	}

	void SetPosition(Board board, uint position, uint8_t val)
	{
		uint bytepos = position / 4;
		uint inbytepos = 3 - (position % 4);
		uint8_t byte = (val << (inbytepos * 2));
		board[bytepos] &= ~(3 << (inbytepos * 2));
		board[bytepos] |= (int)byte;
	}

	void SetPosition(Board board, uint col, uint row, uint8_t val)
	{
		SetPosition(board, xy2c(col, row), val);
	}

	uint8_t OpositeCol(uint8_t col)
	{
		if (col < 2)
			return 0;
		if (col == 2)
			return 3;
		if (col == 3)
			return 2;
		return 0;
	}

	uint8_t GetPosition(Board board, uint position)
	{
		uint bytepos = position / 4;
		uint inbytepos = 3 - (position % 4);
		uint8_t val = board[bytepos] >> (inbytepos * 2);
		return val & 0b11;
	}
	uint8_t GetPosition(Board board, uint col, uint row)
	{
		return GetPosition(board, xy2c(col, row));
	}

	Move ComposeMove(uint col, uint row, uint8_t color, uint8_t isLastMove, uint8_t isPass)
	{
		Move mv = xy2c(col, row) << (16 - POSBITS);
		mv |= (isPass & 1) << 2;
		mv |= (color & 1) << 1;
		mv |= (isLastMove & 1);
		return mv;
	}
	Move SetLastMoveFlag(Move mv, uint8_t lastMoveFlag)
	{
		return (mv & (~1)) | (lastMoveFlag & 1);
	}
	void DecomposeMove(Move move, uint &col, uint &row, uint8_t &color, uint8_t &isLastMove, uint8_t &isPass)
	{
		uint pos = move >> (16 - POSBITS);
		c2xy(pos, col, row);
		isPass = (move >> 2) & 1;
		color = (move >> 1) & 1;
		isLastMove = move & 1;
	}
	void PrintMove(Move mv)
	{
		uint col, row;
		uint8_t color, last, pass;
		DecomposeMove(mv, col, row, color, last, pass);
		std::cout << col << ", " << (char)('a' + row) << "   " << xy2c(col, row) << " color: " << (uint)color << " last: " << (uint)last << " passflag: " << (uint)pass << " \n";
	}

	uint xy2c(uint col, uint row)
	{
		uint pos = row * BOARDSIZE + col;
		return pos;
	}

	void c2xy(uint pos, uint &col, uint &row)
	{
		row = pos / BOARDSIZE;
		col = pos % BOARDSIZE;
	}
}