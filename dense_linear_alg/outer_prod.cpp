#include "mpi.h"
#include <vector>
#include <numeric>

int* outer_product0(int n, int N, std::vector<int>& x, std::vector<int>& y)
{
    std::vector<int> y_global(N);
    MPI_Allgather(y.data(), n, MPI_INT, y_global.data(), n, MPI_INT, MPI_COMM_WORLD);

    int* z = new int[n*N];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < N; j++)
            z[i*N + j] = x[i]*y_global[j];
    return z;
}

int* outer_product1(int n, int N, std::vector<int>& x, std::vector<int>& y)
{
    int my_id, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int* z = new int[n*N];

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
                z[i*N+j+first_col] = x[i]*y[j];

        proc++;
        if (proc == num_procs)
            proc = 0;

        MPI_Isend(y.data(), n, MPI_INT, send_proc, 0, MPI_COMM_WORLD, &send_req);
        MPI_Irecv(tmp.data(), n, MPI_INT, recv_proc, 0, MPI_COMM_WORLD, &recv_req);
        MPI_Wait(&send_req, MPI_STATUS_IGNORE);
        MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
        std::copy(tmp.begin(), tmp.end(), y.begin());
    }

    return z;
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

    std::vector<int> x(n);
    std::vector<int> y(n);

    std::iota(x.begin(), x.end(), my_id*n);
    std::iota(y.begin(), y.end(), my_id*n);
    int* z_global = new int[N*N];

    int* z = outer_product1(n, N, x, y);
    MPI_Allgather(z, n*N, MPI_INT, z_global, n*N, MPI_INT, MPI_COMM_WORLD);
    if (my_id == 0)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                printf("z[%d][%d] = %d\n", i, j, z_global[i*N+j]);
    delete[] z;

    MPI_Finalize();
    return 0;
}
