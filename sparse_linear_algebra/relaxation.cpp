#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "raptor.hpp"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int iter;

    ParCSRMatrix* A;
    ParVector x;
    ParVector b;

    double t0, tfinal;
    double comm_t = 0;

    if (argc <= 1)
    {
        if (rank == 0) printf("Need Matrix Path as Command Line Arg\n");
        MPI_Finalize();
        return 0;
    }

    const char* file = argv[1];
    A = readParMatrix(file);
    x = ParVector(A->global_num_cols, A->on_proc_num_cols);
    b = ParVector(A->global_num_rows, A->local_num_rows);
    x.set_rand_values();
    A->mult(x, b);
    ParVector tmp = ParVector(A->global_num_rows, A->local_num_rows);


    int n_iter = 5;
    double bnorm = b.norm(2);
  
    // Jacobi
    x.set_const_value(1.0);
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    jacobi(A, x, b, tmp, n_iter);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Jacobi Time : %e\n", t0);
    A->residual(x, b, tmp);
    double norm = tmp.norm(2);
    if (rank == 0) printf("Relative Residual %e\n\n", norm/bnorm);

    // Hybrid Jacobi/GS
    x.set_const_value(1.0);
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    ssor(A, x, b, tmp, n_iter);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Hybrid Jacobi Time : %e\n", t0);

    A->residual(x, b, tmp);
    norm = tmp.norm(2);
    if (rank == 0) printf("Relative Residual %e\n", norm/bnorm);

    delete A;

    MPI_Finalize();
    return 0;
}
