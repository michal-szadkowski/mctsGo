
#include <iostream>
#include <stdio.h>
#include <signal.h>
#include <cmath>
#include <time.h>
#include <iomanip>

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <curand.h>
#include <curand_kernel.h>

#include "board.cuh"
#include "comms.cuh"

#define THREADS 1024
#define MAXBLOCKS 82
#define CVALUE 0.3
#define TIMEMAX 30
#define SIMUL 1

void sig_handler(int signo)
{
	if (signo == SIGINT)
	{
		unlink(FSB);
		unlink(FBS);
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

struct TreeNode
{
	brd::Move moveMade;
	brd::Board position;
	uint visits;
	uint black;
	uint white;
	uint depth;
	uint childrenCount;
	TreeNode *children[BOARDSIZE * BOARDSIZE + 1];
	TreeNode *parent;
};

double Value(int wins, int visits, int parentvis, double cst)
{

	if (visits == 0 || parentvis == 0)
		return __DBL_MAX__ / 100;
	double val = (double)wins / (double)visits;
	val += cst * std::sqrt((std::log(parentvis) / (double)visits));
	return val;
}
void PrintNode(TreeNode *node)
{
	brd::PrintBoard(node->position);
	std::cout << node->depth << " " << node->black << " " << node->white << " " << node->visits << "   \n";
	for (int i = 0; i < node->childrenCount; i++)
	{
		TreeNode *child = node->children[i];
		uint8_t color = (brd::GetPlayer(node->position));
		uint wins;
		if (color == BLACK)
		{
			wins = child->black;
		}
		else
			wins = child->white;

		std::cout << i << "  ";
		brd::PrintMove(node->children[i]->moveMade);
		std::cout << "  " << Value(wins, child->visits, node->visits, CVALUE)
				  << " score " << wins << " / " << child->visits << " = " << (double)wins / (double)child->visits << "\n";
	}
}

void BackProp(TreeNode *node, uint black, uint white, uint visits)
{
	while (node != nullptr)
	{
		node->black += black;
		node->white += white;
		node->visits += visits;
		node = node->parent;
	}
}

int IsTerminal(TreeNode *node)
{
	if (node->childrenCount == 0)
	{
		return 1;
	}
	return 0;
}

TreeNode *BestChild(TreeNode *node)
{
	TreeNode *bestChild;
	double bestVal = -1;
	for (int i = 0; i < node->childrenCount; i++)
	{
		TreeNode *child = node->children[i];
		uint8_t color = (brd::GetPlayer(node->position));
		uint wins;
		if (color == BLACK)
		{
			wins = child->black;
		}
		else
			wins = child->white;

		double childVal = Value(wins, child->visits, node->visits, CVALUE);
		if (childVal >= bestVal)
		{
			bestVal = childVal;
			bestChild = child;
		}
	}

	return bestChild;
}
brd::Move MostVisitedChildsMove(TreeNode *node)
{
	brd::Move mv;
	double mval = -1;
	for (int i = 0; i < node->childrenCount; i++)
	{
		TreeNode *child = node->children[i];
		uint8_t color = (brd::GetPlayer(node->position));
		uint wins;
		if (color == BLACK)
		{
			wins = child->black;
		}
		else
			wins = child->white;
		double childVal = (double)wins / child->visits;
		if (child->visits == 0)
			childVal = 0;
		if (childVal >= mval)
		{
			mval = childVal;
			mv = node->children[i]->moveMade;
		}
	}
	std::cout << "Confidence: " << mval << "\n";
	return mv;
}
__global__ void CurandInit(curandState *curand, int count)
{
	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	if (ind < count)
		curand_init(ind, ind, 0, &curand[ind]);
}
__global__ void Simulate(u_char *board, uint *score, curandState *curand)
{

	int ind = threadIdx.x + blockIdx.x * blockDim.x;
	// curand_init(ind, ind, 0, &curand[ind]);

	brd::Board b;
	brd::Copy(board, b);
	int bl, wh;
	int free = brd::GetScores(b, bl, wh);
	brd::Move moves[82];
	int i = 0;
	while (free > 0 && i < MAXMOVES)
	{
		bl = 0;
		wh = 0;
		uint movesCount = brd::GetMoves(b, moves, 82);
		uint ran = (curand_uniform(&curand[ind]) * movesCount * 2);
		uint selection = ran % movesCount; // rand() % movesCount;
		brd::MakeMove(b, moves[selection]);
		free = brd::GetScores(b, bl, wh);
		i++;
	}
	if (bl > wh + KOMI)
		score[ind] = BLACK;
	else if (bl < wh + KOMI)
		score[ind] = WHITE;
	else
		score[ind] = 2;
	// std::cout << "Simulation " << bl << " " << wh << "\n";
}

TreeNode *Select(TreeNode *head)
{
	TreeNode *node = head;
	while (!IsTerminal(node))
	{
		node = BestChild(node);
	}

	// std::cout << "Selection " << node->depth << "\n";
	//  brd::PrintBoard(node->position);
	return node;
}

void Expand(TreeNode *node)
{
	brd::Move moves[82];
	uint moveCount = brd::GetMoves(node->position, moves, 82);
	for (int i = 0; i < moveCount; i++)
	{
		node->children[i] = new TreeNode;
		node->children[i]->depth = node->depth + 1;
		node->children[i]->moveMade = moves[i];
		brd::Copy(node->position, node->children[i]->position);
		brd::MakeMove(node->children[i]->position, moves[i]);
		node->children[i]->parent = node;
		node->children[i]->black = 0;
		node->children[i]->white = 0;
		node->children[i]->visits = 0;
		node->children[i]->childrenCount = 0;
		node->childrenCount++;
	}
}
void FreeTree(TreeNode *head)
{
	if (head != NULL)
	{
		if (!IsTerminal(head))
		{
			for (int i = 0; i < head->childrenCount; i++)
			{
				FreeTree(head->children[i]);
			}
		}
		delete head;
	}
}

brd::Move GetMove(brd::Board board, uint maxtime, curandState *states)
{
	time_t start, end;

	TreeNode *head = new TreeNode;
	if (head == nullptr)
		ERR("new");
	brd::Copy(board, head->position);
	head->black = 0;
	head->white = 0;
	head->visits = 0;
	head->depth = 0;
	head->childrenCount = 0;
	head->parent = nullptr;

	uint black, white, draws, vis;
	TreeNode *selection;
	int i = 0;
	ulong nodes = 0;
	u_char *d_board;
	cudaMalloc(&d_board, BOARDBYTES * MAXBLOCKS);

	time(&start);
	time(&end);
	while (difftime(end, start) < maxtime)
	{
		black = 0;
		white = 0;
		vis = 0;
		draws = 0;

		selection = Select(head);
		Expand(selection);

		for (int j = 0; j < selection->childrenCount; j++)
		{
			cudaMemcpy(d_board + j * BOARDBYTES, selection->children[j]->position, BOARDBYTES, cudaMemcpyHostToDevice);
		}

		thrust::device_vector<uint> results(THREADS * MAXBLOCKS, 2); // 0 for BLACK 1 for WHITE 2 for DRAW
		nodes += selection->childrenCount * THREADS;
		Simulate<<<selection->childrenCount, THREADS>>>(d_board, thrust::raw_pointer_cast(results.data()), states);

		cudaDeviceSynchronize();
		for (int j = 0; j < selection->childrenCount; j++)
		{

			black = 0;
			white = 0;
			vis = 0;
			draws = 0;
			black = thrust::count(results.begin() + j * THREADS, results.begin() + (j + 1) * THREADS, 0);
			white = thrust::count(results.begin() + j * THREADS, results.begin() + (j + 1) * THREADS, 1);
			draws = thrust::count(results.begin() + j * THREADS, results.begin() + (j + 1) * THREADS, 2);
			// std::cout<<"\n\n\n" << draws << "\n\n\n";
			vis += black + white + draws;
			BackProp(selection->children[j], black, white, vis);
		}
		i++;
		time(&end);
	}
	cudaFree(d_board);
	PrintNode(head);
	brd::Move mv = MostVisitedChildsMove(head);
	std::cout << (double)(nodes) / maxtime << " nodes/sec\n";
	FreeTree(head);
	return mv;
}

void Action(int fifos, int fifoc, int color, uint maxtime)
{

	char type;
	brd::Board board;
	brd::Move move;

	curandState *dev_random;
	cudaMalloc((void **)&dev_random, MAXBLOCKS * THREADS * sizeof(curandState));
	CurandInit<<<MAXBLOCKS, THREADS>>>(dev_random, MAXBLOCKS * THREADS);

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

		move = GetMove(board, maxtime, dev_random);

		com::PostMove(move, fifoc);
		brd::MakeMove(board, move);
		// system("clear");
		std::cout << "\n";
		brd::PrintBoard(board);
		brd::PrintMove(move);
	}
}

int main(int argc, char **argv)
{
	signal(SIGINT, sig_handler);

	srand(time(NULL));
	uint8_t color;
	uint maxtime = TIMEMAX;

	if (argc < 2)
		return 0;
	else
	{
		if (argv[1][0] == 'b')
			color = BLACK;
		else if (argv[1][0] == 'w')
			color = WHITE;
		else
			return 0;
		if (argc >= 3)
		{
			if ((maxtime = atoi(argv[2])) <= 0)
				maxtime = TIMEMAX;
		}
	}
	std::cout << maxtime << "\n";
	int fifos, fifoc;
	if (color == BLACK)
		com::ConnectBlack(fifos, fifoc);
	else
		com::ConnectWhite(fifos, fifoc);

	brd::Board board;
	brd::Start(board);
	Action(fifos, fifoc, color, maxtime);

	// curandState *dev_random;
	// cudaMalloc((void **)&dev_random, MAXBLOCKS * THREADS * sizeof(curandState));
	// CurandInit<<<MAXBLOCKS, THREADS>>>(dev_random, MAXBLOCKS * THREADS);
	// GetMove(board, 35, dev_random);
	return 0;
}