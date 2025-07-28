# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc
# For CUDA compile 
NVCC = nvcc

# compile the Cuda version
.PHONY: runcu
run: runcu.cu
	$(NVCC)  -O3 -o runcu runcu.cu -lm
# compile cublas included
.PHONY: runcublas                                                     
run: runcu.cu                                                              $(NVCC)  -O3 -DUSE_CUBLAS -o runcublas runcu.cu -lm -lcublas

# =========================
# The below is not used hree.


# the most basic way of building that is most likely to work on most systems
.PHONY: run
run: run.c
	$(CC) -O3 -o run run.c -lm

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin -n 3
rundebug: run.c
	$(CC) -g -o run run.c -lm

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://simonbyrne.github.io/notes/fastmath/
# -Ofast enables all -O3 optimizations.
# Disregards strict standards compliance.
# It also enables optimizations that are not valid for all standard-compliant programs.
# It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific
# -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens.
# It turns off -fsemantic-interposition.
# In our specific application this is *probably* okay to use
#.PHONY: run
#runfast: run.c
#	$(CC) -O3 -o run -fopenmp -march=native run.c -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./run out/model.bin
.PHONY: runomp
runomp: run.c
	$(CC) -O3 -fopenmp -march=native run.c  -lm  -o run


.PHONY: clean
clean:
	rm -f run
