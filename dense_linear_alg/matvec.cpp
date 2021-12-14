#include "mpi.h"
#include <stdlib.h>
#include <vector>
#include <numeric>

int* matvec0(int n, int N, std::vector<int>& A, std::vector<int>& x)
{
    std::vector<int> x_global(N);
    MPI_Allgather(x.data(), n, MPI_INT, x_global.data(), n, MPI_INT, MPI_COMM_WORLD);

    int* y = new int[n];
    for (int i = 0; i < n; i++)
    {
        y[i] = 0;
        for (int j = 0; j < N; j++)
            y[i] += A[i*N+j]*x_global[j];
    }
    return y;
}

int* matvec1(int n, int N, std::vector<int>& A, std::vector<int>& x)
{
    int my_id, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int* y = new int[n];
    for (int i = 0; i < n; i++)
        y[i] = 0;

    int proc = my_id;
    int first_col;
    int send_proc = my_id - 1;
    if (send_proc < 0) send_proc += num_procs;
    int recv_proc = my_id + 1;
    if (recv_proc == num_procs) recv_proc -= num_procs;
    std::vector<int> tmp(n);

    MPI_Request send_req, recv_req;

    for (int i = 0; i < num_procs; i++)
    {
        first_col = n*proc;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                y[i] += A[i*N+first_col+j]*x[j];

        proc++;
        if (proc == num_procs)
            proc = 0;

        MPI_Isend(x.data(), n, MPI_INT, send_proc, 0, MPI_COMM_WORLD, &send_req);
        MPI_Irecv(tmp.data(), n, MPI_INT, recv_proc, 0, MPI_COMM_WORLD, &recv_req);
        MPI_Wait(&send_req, MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
        std::copy(tmp.begin(), tmp.end(), x.begin());
    }

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
    int n = N / num_procs;

    std::vector<int> A(n*N);
    std::vector<int> x(n);
    int first_row = my_id*n;
    int first_nnz = first_row*N;

    for (int i = 0; i < n; i++)
    {
        x[i] = first_row + i;
        for (int j = 0; j < N; j++)
        {
            A[i*N+j] = first_nnz + i*N + j;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    int* y = matvec1(n, N, A, x);
    double tfinal = MPI_Wtime() - t0;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (my_id == 0) printf("Matvec Time %e\n", t0);

    delete[] y;

    MPI_Finalize();
    return 0;
}
