#include "comms.h"
#include "board.h"

namespace com
{
	void ServerMakeFifo(int &fifosb, int &fifobs, int &fifosw, int &fifows)
	{
		unlink(FSB);
		unlink(FBS);
		unlink(FSW);
		unlink(FWS);
		mkfifo(FSB, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
		mkfifo(FBS, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
		mkfifo(FSW, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
		mkfifo(FWS, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
		std::cout << "Awaiting connection\n";
		fifosb = open(FSB, O_WRONLY);
		fifobs = open(FBS, O_RDONLY);
		std::cout << "Black connected\n";
		fifosw = open(FSW, O_WRONLY);
		fifows = open(FWS, O_RDONLY);
		std::cout << "White connected\n";
	}

	void ConnectBlack(int &fifosb, int &fifobs)
	{
		fifosb = open(FSB, O_RDONLY);
		fifobs = open(FBS, O_WRONLY);
	}

	void ConnectWhite(int &fifosw, int &fifows)
	{
		fifosw = open(FSW, O_RDONLY);
		fifows = open(FWS, O_WRONLY);
	}

	void PostMove(brd::Move mv, int fifo)
	{
		char *buf = (char *)malloc(3);
		buf[0] = MSG_MOVE;
		memcpy(buf + 1, &mv, 2);
		write(fifo, buf, 3);
		free(buf);
	}
	void PostStart(int fifo)
	{
		char *buf = (char *)malloc(3);
		buf[0] = MSG_START;
		buf[1] = 1;
		buf[2] = 1;
		write(fifo, buf, 3);
		free(buf);
	}
	void PostEnd(int fifo)
	{
		char *buf = (char *)malloc(3);
		buf[0] = MSG_STOP;
		buf[1] = 1;
		buf[2] = 1;
		write(fifo, buf, 3);
		free(buf);
	}

	void ReadF(int fifo, char &type, brd::Move &mv)
	{
		char *buf = (char *)malloc(3);

		read(fifo, buf, 3);

		if (buf[0] == MSG_START || buf[0] == MSG_MOVE || buf[0] == MSG_STOP)
			type = buf[0];
		else
			type = MSG_ERR;
			
		if (type == MSG_MOVE)
			memcpy(&mv, buf + 1, 2);
		else
			memset(&mv, 0, 2);
		free(buf);
	}
}