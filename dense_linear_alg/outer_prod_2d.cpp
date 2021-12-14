#include "mpi.h"
#include <vector>
#include <numeric>
#include <cmath>

int* outer_product0(int n, int p_x, int p_y, MPI_Comm row_comm, MPI_Comm col_comm, std::vector<int>& x, std::vector<int>& y)
{
    // 1. Broadcast x
    // 2. Broadcast y
    // 3. multiply

    MPI_Bcast(x.data(), n, MPI_INT, p_y, row_comm);
    MPI_Bcast(y.data(), n, MPI_INT, p_x, col_comm);
    
    int* z = new int[n*n];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            z[i*n + j] = x[i]*y[j];
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

    std::vector<int> x(n);
    std::vector<int> y(n);

    if (p_x == p_y) // diagonal
    {
        std::iota(x.begin(), x.end(), p_x*n);
        std::iota(y.begin(), y.end(), p_y*n);
    }

    MPI_Comm row_comm;
    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, p_y, my_id, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, p_x, my_id, &col_comm);

    int* z_global = new int[N*N];

    int* z = outer_product0(n, p_x, p_y, row_comm, col_comm, x, y);

    MPI_Allgather(z, n*n, MPI_INT, z_global, n*n, MPI_INT, MPI_COMM_WORLD);
    int row, col, row_group_start, col_group_start;
    if (my_id == 0)
    {
        for (int y = 0; y < sqrt_p; y++)
        {
            for (int i = 0; i < n; i++)
            {
                row = y*n + i;
                row_group_start = y*sqrt_p*n*n;
                for (int x = 0; x < sqrt_p; x++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        col = x*n + j;
                        col_group_start = x*n*n;
                        printf("z[%d][%d] = %d\n", row, col, z_global[row_group_start + col_group_start + i*n + j]);
                    }
                }
            }
        }   
    }

    delete[] z;

    delete[] z_global;

    MPI_Finalize();
    return 0;
}
