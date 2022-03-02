#ifndef __COMMS
#define __COMMS

#include <cstdio>
#include <iostream>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctype.h>

#include "board.cuh"

#define FSB "/tmp/fifosb.fifo"
#define FBS "/tmp/fifobs.fifo"
#define FSW "/tmp/fifosw.fifo"
#define FWS "/tmp/fifows.fifo"

#define MSG_MOVE 'm'
#define MSG_START 's'
#define MSG_STOP 'e'
#define MSG_ERR 'q'

namespace com
{
	void ServerMakeFifo(int &fifosb, int &fifobs, int &fifosw, int &fifows);

	void ConnectBlack(int &fifosb, int &fifobs);
	void ConnectWhite(int &fifosw, int &fifows);
	void PostMove(brd::Move mv, int fifo);
	void PostStart(int fifo);
	void PostEnd(int fifo);
	void ReadF(int fifo, char &type, brd::Move &mv);
}

#endif