#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include "raptor.hpp"
#include "util/linalg/external/parmetis_wrapper.hpp"

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
    int n_iter = 1000;

    // Block Mapping SpMV
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int iter = 0; iter < n_iter; iter++)
        A->mult(x, b);
    tfinal = (MPI_Wtime() - t0) / n_iter;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Block Mapping SpMV Time : %e\n", t0);


    // Partition (with ParMetis)
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    int* parts = parmetis_partition(A);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("ParMETIS Partitioning Time : %e\n", t0);

    // Re-Map Matrix
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    std::vector<int> new_local_rows;
    ParCSRMatrix* A_part = repartition_matrix(A, parts, new_local_rows);
    ParVector x_part = ParVector(A_part->global_num_cols, A_part->on_proc_num_cols);
    ParVector b_part = ParVector(A_part->global_num_rows, A_part->local_num_rows);
    tfinal = MPI_Wtime() - t0;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("ReMapping Matrix to Processes Time : %e\n", t0);

    // Repartitioned SpMV
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int iter = 0; iter < n_iter; iter++)
        A_part->mult(x_part, b_part);
    tfinal = (MPI_Wtime() - t0) / n_iter;
    MPI_Reduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Repartitioned SpMV Time : %e\n", t0);


    delete[] parts;
    delete A;

    MPI_Finalize();
    return 0;
}
