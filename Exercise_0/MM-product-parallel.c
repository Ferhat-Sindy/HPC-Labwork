/******************************************************************************
* FILE: mm.c
* DESCRIPTION:  
*   This program calculates the product of matrix a[nra][nca] and b[nca][ncb],
*   the result is stored in matrix c[nra][ncb].
*   The max dimension of the matrix is constraint with static array declaration,
*   for a larger matrix you may consider dynamic allocation of the arrays, 
*   but it makes a parallel code much more complicated (think of communication),
*   so this is only optional.
*   
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

// #define NRA 1000                 /* number of rows in matrix A */
// #define NCA 1000                 /* number of columns in matrix A */
// #define NCB 1000                  /* number of columns in matrix B */
#define N 1000

int main (int argc, char *argv[]) 
{
    int np, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if(rank == 0){
        int i, j, k;
        /* for simplicity, set NRA=NCA=NCB=N  */
        
        int NRA=N;
        int NCA=N;
        int NCB=N;
    
        double  a[NRA][NCA],           /* matrix A to be multiplied */
                b[NCA][NCB],           /* matrix B to be multiplied */
                c[NRA][NCB];           /* result matrix C */
    
    
      /*** Initialize matrices ***/
      
        for (i=0; i<NRA; i++)
            for (j=0; j<NCA; j++)
                a[i][j]= i+j;
      
        for (i=0; i<NCA; i++)
            for (j=0; j<NCB; j++)
                b[i][j]= i*j;
      
        for (i=0; i<NRA; i++)
            for (j=0; j<NCB; j++)
              c[i][j]= 0;
        
        int f;
        int ROW = NCA/(np-1);
        int ROW_rem = NCA%(np-1);
        
        double startTime, endTime;
        
        startTime = MPI_Wtime();
        
        for(f = 1; f < np; f++){
            MPI_Send(&a[0+ROW*(f-1)][0], NCA*ROW, MPI_DOUBLE, f, 0,MPI_COMM_WORLD);
            MPI_Send(&b[0][0], NCA*NCB, MPI_DOUBLE, f, 1,MPI_COMM_WORLD);
        }
        
        /* Solving the remainder rows on main processor*/
        
        for (i=NCB - ROW_rem; i<NCB; i++)    
        {
            for(j=0; j<NCB; j++)       
                for (k=0; k<NCA; k++)
                    c[i][j] += a[i][k] * b[k][j];
        }
        
        
        for(f = 1; f < np; f++){
            MPI_Recv(&c[0+ROW*(f-1)][0], ROW*NCB, MPI_DOUBLE, f, 2,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        endTime = MPI_Wtime();
        
        printf("MM-product-parallel took %f seconds\n", (endTime - startTime)/10);
        
        /*for (i=0; i<NRA; i++)    
        {
            printf("\n");
            for(j=0; j<NCB; j++){
                printf("%f ", c[i][j]);
            }
        }*/ 

    }

    if (rank != 0) {
        
        int NCA = N;
        int NCB = N;
        int ROW = NCA/(np-1);
        int i,j,k;
        double a_temp[ROW][NCA];
        double b_temp[NCA][NCB];
        double c_temp[ROW][NCB];

        MPI_Recv(&a_temp[0], NCA*ROW, MPI_DOUBLE, 0, 0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&b_temp[0], NCA*NCB, MPI_DOUBLE, 0, 1,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        
        for (i=0; i<ROW; i++)    
        {
            for(j=0; j<NCB; j++)       
                for (k=0; k<NCA; k++)
                    c_temp[i][j] += a_temp[i][k] * b_temp[k][j];
        }

        MPI_Send(&c_temp[0][0], ROW*NCB, MPI_DOUBLE, 0, 2,MPI_COMM_WORLD);
    }
    // Finalize MPI
    MPI_Finalize();
    return 0; 
}
