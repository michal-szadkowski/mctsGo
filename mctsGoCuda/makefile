appname := CudaGo

NVCC := nvcc
NVCC_FLAGS := -lcudart -dc -I /usr/local/cuda/include --ptxas-options=-v -g -G
CC := g++
CC_FLAGS := -Wall -g

#srcfiles := $(shell find . -maxdepth 1 -name "*.cpp")
directory := ../bin
srcfiles := board.cu CudaGo.cu comms.cu
objects  := $(patsubst %.cpp, %.o, $(srcfiles))
objects := $(patsubst %.cu, %.o, $(objects))

all: $(directory) $(appname) clean

$(appname): $(objects)
	$(NVCC) $(NVCCFLAGS) -o $(appname) $(objects)
	mv $(appname) ../bin

%.o : %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@

%.o : %.cpp %.h
	$(CC) $(CC_FLAGS) -c $< -o $@

%.o : %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

$(directory):
	mkdir -p $@

clean:
	rm -f $(objects)

