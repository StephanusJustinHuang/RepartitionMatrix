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

    int n = 5;
    int system = 0;
    double strong_threshold = 0.25;
    int iter;
    int num_variables = 1;

    coarsen_t coarsen_type = PMIS;
    interp_t interp_type = Extended;
    int hyp_coarsen_type = 8; // PMIS
    int hyp_interp_type = 6; // Extended
    int p_max_elmts = 0;
    int agg_num_levels = 0;

    ParMultilevel* ml;
    ParCSRMatrix* A;
    ParVector x;
    ParVector b;

    double t0, tfinal;

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
    x.set_const_value(0.0);

    // Preconditioned Conjugate Gradient
    ParVector sas_sol = ParVector(x);
    std::vector<double> res;
    double precond_t = 0;
    double comm_t = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    ml = new ParRugeStubenSolver(strong_threshold);
    ml->setup(A);
    PCG(A, ml, sas_sol, b, res, 1e-07, -1, &precond_t, &comm_t);
    iter = res.size() - 1;
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("PCG Time: %e\n", t0);
    MPI_Reduce(&precond_t, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Preconditioner Time: %e\n", t0);
    if (rank == 0) 
    {
        printf("Solved in %d iterations\n\n\n", iter);
    }
    delete ml;


    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    std::vector<double> new_res;
    comm_t = 0;
    CG(A, x, b, new_res, 1e-07, -1, &comm_t);
    iter = new_res.size() - 1;
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("CG Time: %e\n", t0);
    if (rank == 0) 
    {
        printf("Solved in %d iterations\n", iter);
    }

    delete A;

    MPI_Finalize();
    return 0;
}
