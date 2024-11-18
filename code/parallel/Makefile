# Number of processes:
N=2
HOSTFILE = hostfile 
# Number of thread 
Number_of_OMP_THREADS = 2


Number_of_OMP_THREADSOPT = 2


main: main.c lib/libbmp/libbmp.c lib/bmp.c
	mpicc -o $@ $^ -fopenmp -lm -Wall -g

mainopt: mainopt.c lib/libbmp/libbmp.c lib/bmp.c
	mpicc -o $@ $^ -fopenmp -lm -Wall -g

run: main
	OMP_PROC_BIND=true OMP_PLACES=cores mpiexec -bind-to none -n $(N) -hostfile $(HOSTFILE) -x OMP_NUM_THREADS=$(Number_of_OMP_THREADS) ./main

runo: mainopt
	OMP_PROC_BIND=true OMP_PLACES=cores mpiexec -bind-to none -n $(N) -hostfile $(HOSTFILE) -x OMP_NUM_THREADS=$(Number_of_OMP_THREADSOPT) ./mainopt

clean:
	@rm main boxblur.bmp sobel.bmp

cleano: 
	@rm mainopt boxblur.bmp sobel.bmp

.PHONY: clean
