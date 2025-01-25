CC=g++
CFLAGS= -std=c++17 -Wall -g -fno-omit-frame-pointer -O3 -funroll-loops -fassociative-math  -march=native -mcpu=native -ffast-math -fopenmp 

all: BSM1 BSM3-5 BSM6 BSM6 BSM6_arm BSM7 BSM8
mpi: BSM4 BSM4-5

BSM: BSM.cxx
	$(CC) BSM.cxx -o BSM

BSM1: BSM.cxx
	$(CC) $(CFLAGS) BSM.cxx -o BSM1

BSM2: BSM2.cxx
	$(CC) $(CFLAGS) BSM2.cxx -o BSM2

BSM3: BSM3.cxx
	$(CC) $(CFLAGS) -fopenmp BSM3.cxx -o BSM3

BSM4: BSM4.cxx
	mpic++ $(CFLAGS) -o BSM4 BSM4.cxx

BSM4-5: BSM4-5.cxx
	mpic++ $(CFLAGS) -o BSM4-5 BSM4-5.cxx

BSM5: BSM5.cxx
	$(CC) $(CFLAGS) -fopenmp BSM5.cxx -o BSM5

BSM5-5: BSM5-5.cxx
	$(CC) $(CFLAGS) -fopenmp -mcpu=native BSM5-5.cxx -o BSM5-5

BSM3-5: BSM3-5.cxx
	$(CC) $(CFLAGS) -fopenmp BSM3-5.cxx -o BSM3-5

BSM6: BSM6.cxx
	$(CC) $(CFLAGS) -fopenmp BSM6.cxx -o BSM6

BSM6_arm: BSM6.cxx
	armclang++ $(CFLAGS)  -o BSM6_arm BSM6.cxx

BSM7: BSM7.cxx
	armclang++ $(CFLAGS) -o BSM7 BSM7.cxx

BSM8: BSM8.cxx
	armclang++ $(CFLAGS) -lboost_system -lboost_random -fopenmp -o BSM8 BSM8.cxx 

BSM9: BSM9.cxx
	armclang++ $(CFLAGS) -lboost_system -lboost_random -o BSM9 BSM9.cxx 

BSMTest: BSMTest.cxx
	armclang++ $(CFLAGS) -lboost_system -lboost_random -o BSMTest BSMTest.cxx
BSM8test:BSM8test.cxx
	armclang++ $(CFLAGS) -lm -lboost_system -lboost_random -o BSM8test BSM8test.cxx

clean:
	rm -f BSM BSM2 BSM3 BSM4 BSM5 BSM3-5 BSM4-5 BSM5-5 BSM6 BSM6_arm BSM7 BSM8 BSM9 BSMTest BSM8test
