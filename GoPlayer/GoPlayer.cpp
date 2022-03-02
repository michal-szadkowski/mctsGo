#include <iostream>
#include <stdio.h>
#include <signal.h>
#include <string>
#include "board.h"
#include "comms.h"
#include <stdlib.h>
#include <time.h>
#include <limits>

void sig_handler(int signo)
{
	if (signo == SIGINT)
	{
		unlink(FSB);
		unlink(FBS);
		unlink(FSW);
		unlink(FWS);
	}

	exit(0);
}
void quit(std::string s)
{
	std::cout << s << "\n";
	unlink(FSB);
	unlink(FBS);
	unlink(FSW);
	unlink(FWS);
	exit(0);
}

brd::Move GetMove(brd::Board board, uint8_t color)
{
	uint col, row;
	std::string in;
	do
	{
		std::cout << "Input move: ";
		std::cin >> in;
		if (in.compare("p") == 0)
		{
			col = 0;
			row = 0;
			return brd::ComposeMove(0, 0, color, 0, 1);
		}
		if (in.length() > 3 || in.length() < 2)
			continue;
		col = in[0] - '0';
		row = in[1] - 'a';

	} while (brd::IsLegal(board, col, row, color) != 1);
	return brd::ComposeMove(col, row, color, 0, 0);
}

void ClientAction(int fifos, int fifoc, uint8_t color)
{
	char type;
	brd::Board board;
	brd::Move move;

	brd::Start(board);
	while (true)
	{
		type = 0;
		com::ReadF(fifos, type, move);
		if (type == MSG_ERR)
		{
			quit("");
		}
		if (type == MSG_MOVE)
		{
			if (brd::MakeMove(board, move) == 1)
			{
				quit("Read illegal move");
			}
		}
		if (type == MSG_STOP)
		{
			quit("End");
		}
		system("clear");
		brd::PrintBoard(board);
		brd::PrintMove(move);

		move = GetMove(board, color);
		com::PostMove(move, fifoc);
		brd::MakeMove(board, move);
		system("clear");
		brd::PrintBoard(board);
		brd::PrintMove(move);
	}
}

int main(int argc, char **argv)
{
	signal(SIGINT, sig_handler);
	uint8_t color;

	if (argc != 2)
		return 0;
	else
	{
		if (argv[1][0] == 'b')
			color = BLACK;
		else if (argv[1][0] == 'w')
			color = WHITE;
		else
			return 0;
	}

	int fifos, fifoc;
	if (color == BLACK)
		com::ConnectBlack(fifos, fifoc);
	else
		com::ConnectWhite(fifos, fifoc);

	ClientAction(fifos, fifoc, color);

	return 0;
}