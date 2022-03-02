#include <iostream>
#include <stdio.h>
#include <signal.h>
#include "board.h"
#include "comms.h"
#include <stdlib.h>
#include <time.h>

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

int CheckForEnd(int fifob, int fifow, brd::Board board)
{
	int white, black;
	int free = brd::GetScores(board, black, white);
	std::cout << "\nblack: " << black << " white:" << white << "\n";
	if (free == 0)
	{

		com::PostEnd(fifob);
		com::PostEnd(fifow);
		if (black > white + KOMI)
			return BLACK;
		else if (white + KOMI > black)
			return WHITE;
		return 2;
	}
	return -1;
}

void ReciveAndPost(int fifor, int fifop, brd::Move &mv, brd::Board board)
{
	char type;
	brd::Move move;
	com::ReadF(fifor, type, move);
	if (type != MSG_MOVE)
		quit("Wrong message recived\n");
	if (brd::MakeMove(board, move) == 1)
	{
		quit("Illegal move recived\n");
	};
	com::PostMove(move, fifop);
}

void Action(int fifosb, int fifobs, int fifosw, int fifows)
{
	brd::Board board;
	brd::Start(board);
	brd::Move move;
	int win;

	com::PostStart(fifosb);
	while (true)
	{
		system("clear");
		brd::PrintBoard(board);

		ReciveAndPost(fifobs, fifosw, move, board);

		if ((win = CheckForEnd(fifosb, fifosw, board)) >= 0)
		{
			system("clear");
			brd::PrintBoard(board);
			int white, black;
			brd::GetScores(board, black, white);
			std::cout << "\nblack: " << black << " white:" << white << "\n";
			if (win == WHITE)
				std::cout << "white (green) won\n";
			else if (win == BLACK)
				std::cout << "black (red) won\n";
			else
				std::cout << "draw\n";
			quit("Game finished\n");
		}
		system("clear");
		brd::PrintBoard(board);

		ReciveAndPost(fifows, fifosb, move, board);
		if ((win = CheckForEnd(fifosb, fifosw, board)) >= 0)
		{
			system("clear");
			brd::PrintBoard(board);
			int white, black;
			brd::GetScores(board, black, white);
			std::cout << "\nblack: " << black << " white:" << white << "\n";
			if (win == WHITE)
				std::cout << "white (green) won\n";
			else if (win == BLACK)
				std::cout << "black (red) won\n";
			else
				std::cout << "draw\n";
			quit("Game finished\n");
		}
	}
}

int main(int argc, char **argv)
{
	signal(SIGINT, sig_handler);
	// srand(time(NULL));
	// brd::Board board;
	// brd::Start(board);

	// brd::Move mvs[82];
	// brd::Move move;
	// unsigned long i = 0;
	// int twopassflag = 0;
	// int a, b;
	// while (twopassflag < 2)
	// {
	// 	brd::PrintBoard(board);
	// 	uint count = brd::GetMoves(board, mvs, 82);
	// 	std::cout << count << " " << i << "  " << brd::GetScores(board, a, b) << "\n";
	// 	move = mvs[rand() % count];
	// 	if (count > 1)
	// 	{
	// 		twopassflag = 0;
	// 	}
	// 	else
	// 	{
	// 		twopassflag++;
	// 	}
	// 	brd::PrintMove(move);
	// 	brd::MakeMove(board, move);
	// 	i++;
	// }
	// brd::GetScores(board, a, b);
	// std::cout << "\n"
	// 		  << a << " " << b;

	int fsb, fbs, fsw, fws;
	char type;
	brd::Move mv;
	com::ServerMakeFifo(fsb, fbs, fsw, fws);
	Action(fsb, fbs, fsw, fws);

	unlink(FSW);
	unlink(FWS);
	unlink(FBS);
	unlink(FSB);
	return 0;
}