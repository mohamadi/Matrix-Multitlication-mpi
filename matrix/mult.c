/*
 	CPSC 521: Parallel Algorithms and Architectures
	Assignment 2: Matrix Multitlication
	Author: Hamid Mohamadi, mohamadi@alumni.ubc.ca
*/

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <string.h>

void mulUpdt(int, int, int, int, int *, int *, int *);
void scatter(const char *, int, int, const int, int *);
void simulate(const int, int, int, const int, int *, int *);
void gather(int, int, const int, int *);

int main(int argc, char *argv[]){    
	const int kth=atoi(argv[1]); /* k-th power of matrix*/
	const int rows=atoi(argv[2]); /* Matrix size */
	const char *gName=argv[3]; /* Input file name*/
	
	double time; /* Computation time*/
	int rank, size; 
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
        
	int gran = rows/size;
	int *cgMatrix = (int *)malloc(gran*rows*sizeof(int));
	int *gMatrix = (int *)malloc(gran*rows*sizeof(int));
	
	time = MPI_Wtime();
	/****************************************************************
	1. Scatter: 
	Reading input matrix from file and assigning rows to  processes
	****************************************************************/
	scatter(gName, rank, size, gran, gMatrix);

	/****************************************************************
	2. Simulate: 
	Passing the data of each process to other processes and simulate
	****************************************************************/
	memcpy(cgMatrix, gMatrix, gran*rows*sizeof(int));
	simulate(kth, rank, size, gran, gMatrix, cgMatrix);
	
	/****************************************************************
	3. Gather:
	Writing the the reslut, matrix^k, to matrix_out.txt
	****************************************************************/
	gather(rank, size, gran, gMatrix);

	/* Finalizing*/	
	free(cgMatrix);
	free(gMatrix);
	time = MPI_Wtime() - time;
	if(rank==0)
		printf("Computation time is %.4f seconds\n", time);	
	MPI_Finalize();
	return(0);
}

void mulUpdt(int gran, int size, int rank, int round, int *fPart, int *sPart, int *f){
	int i, j, k, rows = gran*size ;
	int *mulTemp = (int *)malloc(gran*gran*sizeof(int));
	for (i=0; i<gran; i++)
		for (j=0; j<gran; j++){
			mulTemp[i*gran+j] = 0;
			for (k=0; k<rows; k++)
				mulTemp[i*gran+j] += fPart[i*rows+k]*sPart[j*rows+k];
		}
	for (i=0; i<gran; i++)
		for (j=0; j<gran; j++)
			f[i*rows+gran*((round+rank)%size)+j] = mulTemp[i*gran+j];
	free(mulTemp);
}

void scatter(const char *gName, int rank, int size, const int gran, int *gMatrix){
	int i, j, k, rows = gran*size;
	int sendto = (rank + 1) % size;
	int recvfrom = ((rank + size) - 1) % size;
	MPI_Status status;
	int *outbuf = (int *)malloc(gran*rows*sizeof(int));
	
	if(rank==0){
		FILE *gFile; 
		gFile = fopen(gName, "rb");
		for(i=0; i<gran; i++)
			for (j=0; j<rows; j++)
				fscanf(gFile,"%d", gMatrix+i*rows+j);
		for(k=0; k<size-rank-1; k++){
			for(i=0; i<gran; i++)
				for (j=0; j<rows; j++)
					fscanf(gFile,"%d", outbuf+i*rows+j);
			MPI_Send(outbuf, gran*rows, MPI_INT, sendto, 0, MPI_COMM_WORLD);
		}	
		fclose(gFile);
	}	
	else{
		MPI_Recv(outbuf, gran*rows, MPI_INT, recvfrom, 0, MPI_COMM_WORLD, &status);
		memcpy(gMatrix, outbuf, gran*rows*sizeof(int));
		for(k=0; k<size-rank-1; k++){
			MPI_Recv(outbuf, gran*rows, MPI_INT, recvfrom, 0, MPI_COMM_WORLD, &status);
			MPI_Send(outbuf, gran*rows, MPI_INT, sendto, 0, MPI_COMM_WORLD);
		}
	}	
	free(outbuf);
}

void simulate(const int kth, int rank, int size, const int gran, int *gMatrix, int *cgMatrix){
	int t=kth, round, rows=gran*size;
	int sendto = (rank + 1) % size;
	int recvfrom = ((rank + size) - 1) % size;

	MPI_Status status;
	
	int *inbuf = (int *)malloc(gran*rows*sizeof(int));
	int *outbuf = (int *)malloc(gran*rows*sizeof(int));
	int *f = (int *)calloc(gran*rows,sizeof(int));
	
	while(1){
		--t;
		round=size;
		memcpy(outbuf, gMatrix, gran*rows*sizeof(int));
		mulUpdt(gran, size, rank, round, cgMatrix, gMatrix, f);
		while (1) {
			--round;
			if (!(rank % 2)){
				MPI_Send(outbuf, gran*rows, MPI_INT, sendto, 0, MPI_COMM_WORLD);
				MPI_Recv(inbuf, gran*rows, MPI_INT, recvfrom, 0, MPI_COMM_WORLD, &status);						
			}
			else
			{
				MPI_Recv(inbuf, gran*rows, MPI_INT, recvfrom, 0, MPI_COMM_WORLD, &status);
				MPI_Send(outbuf, gran*rows, MPI_INT, sendto, 0, MPI_COMM_WORLD);		
			}
			memcpy(outbuf, inbuf, gran*rows*sizeof(int));
			mulUpdt(gran, size, rank, round, cgMatrix, inbuf, f);
			if (round == 1)	break;
		}
		memcpy(gMatrix, f, gran*rows*sizeof(int));
		if(t==1) break;
	}
	free(inbuf);
	free(outbuf);
	free(f);
}

void gather(int rank, int size, const int gran, int *gMatrix){
	int i, j, k, rows = gran*size;
	int sendto = (rank + 1) % size;
	int recvfrom = ((rank + size) - 1) % size;
	MPI_Status status;	
	int *outbuf = (int *)malloc(gran*rows*sizeof(int));
	
	if (rank != 0){
		memcpy(outbuf, gMatrix, gran*rows*sizeof(int));
		MPI_Send(outbuf, gran*rows, MPI_INT, recvfrom, 0, MPI_COMM_WORLD);
		for(k=0; k<size-rank-1; k++){
			MPI_Recv(outbuf, gran*rows, MPI_INT, sendto, 0, MPI_COMM_WORLD, &status);
			MPI_Send(outbuf, gran*rows, MPI_INT, recvfrom, 0, MPI_COMM_WORLD);
		}	
	}
	else{
		FILE *oFile;
		oFile = fopen("matrix_out.txt", "w");	
		memcpy(outbuf, gMatrix, gran*rows*sizeof(int));
		for(i=0; i<gran; i++){
			for(j=0; j<rows;j++)
				fprintf(oFile,"%d ", outbuf[i*rows+j]);
			fprintf(oFile,"\n");
		}
		for(k=0; k<size-rank-1; k++){
			MPI_Recv(outbuf, gran*rows, MPI_INT, sendto, 0, MPI_COMM_WORLD, &status);
			for(i=0; i<gran; i++){
				for(j=0; j<rows;j++)
					fprintf(oFile,"%d ", outbuf[i*rows+j]);
				fprintf(oFile,"\n");
			}
		}
		fclose(oFile);
	}
	free(outbuf);

}
