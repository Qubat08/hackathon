CC=g++
CFLAGS= -std=c++17 -Wall -g -fno-omit-frame-pointer -O3 -funroll-loops -fassociative-math  -march=native -mcpu=native -ffast-math -fopenmp 

all: BSM BSM_OpenMP BSM_Generator BSM_Generator BSM_Generator_arm BSM_Simd BSM_Boost


BSM: BSM.cxx
	$(CC) BSM.cxx -o BSM

BSM_OpenMP: BSM_OpenMP.cxx
	$(CC) $(CFLAGS) -fopenmp BSM_OpenMP.cxx -o BSM_OpenMP

BSM_Generator: BSM_Generator.cxx
	$(CC) $(CFLAGS) -fopenmp BSM_Generator.cxx -o BSM_Generator

BSM_Generator_arm: BSM_Generator.cxx
	armclang++ $(CFLAGS)  -o BSM_Generator_arm BSM_Generator.cxx

BSM_Simd: BSM_Simd.cxx
	armclang++ $(CFLAGS) -o BSM_Simd BSM_Simd.cxx

BSM_Boost: BSM_Boost.cxx
	armclang++ $(CFLAGS) -lboost_system -lboost_random -fopenmp -o BSM_Boost BSM_Boost.cxx 

clean:
	rm -f BSM BSM_OpenMP BSM_Generator BSM_Generator_arm BSM_Simd BSM_Boost
