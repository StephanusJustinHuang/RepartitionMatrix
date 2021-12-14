#include "mpi.h"
#include <stdlib.h>
#include <vector>
#include <numeric>
#include <cmath>

int* matvec0(int n, int p_x, int p_y, MPI_Comm row_comm, MPI_Comm col_comm, std::vector<int>& A, std::vector<int>& x)
{
    // 1. Broadcast x
    // 2. Broadcast y
    // 3. multiply

    MPI_Bcast(x.data(), n, MPI_INT, p_x, col_comm);
    
    int* y = new int[n];
    for (int i = 0; i < n; i++)
    {
        y[i] = 0;
        for (int j = 0; j < n; j++)
        {
            y[i] += A[i*n+j]*x[j];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, y, n, MPI_INT, MPI_SUM, row_comm);
    return y;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int my_id, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc == 1)
    {
        printf("Requires command line argument for n\n");
        MPI_Finalize();
        return 0;
    }

    int N = (int) atoi(argv[1]);
    int sqrt_p = std::sqrt(num_procs);

    if (sqrt_p * sqrt_p != num_procs)
    {
        printf("Num Procs is not a square\n");
        MPI_Finalize();
        return 0;
    }
    if (N % sqrt_p)
    {
        printf("N does not evenly divide sqrt(p)\n");
        MPI_Finalize();
        return 0;
    }

    int n = N / sqrt_p;
    int p_x = my_id % sqrt_p;
    int p_y = my_id / sqrt_p;

    std::vector<int> A(n*n);
    std::vector<int> x(n);

    if (p_x == p_y) // diagonal
    {
        std::iota(x.begin(), x.end(), p_x*n);
    }

    int first_idx = p_y*n*N;
    int first_col = p_x*n;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i*n+j] = first_idx + i*N + first_col + j;

    MPI_Comm row_comm;
    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, p_y, my_id, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, p_x, my_id, &col_comm);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    int* y = matvec0(n, p_x, p_y, row_comm, col_comm, A, x);
    double tfinal = MPI_Wtime() - t0;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (my_id == 0) printf("2D Matvec Time : %e\n", t0);

    delete[] y;

    MPI_Finalize();
    return 0;
}
