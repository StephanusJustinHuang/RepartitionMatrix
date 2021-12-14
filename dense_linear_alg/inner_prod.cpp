#include "mpi.h"
#include <stdlib.h>
#include <vector>
#include <numeric>

int inner_product(int n, std::vector<int>& x, std::vector<int>& y)
{
    int global_sum;
    int sum = 0;
    for (int i = 0; i < n; i++)
        sum += x[i]*y[i];
    MPI_Allreduce(&sum, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return global_sum;
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

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    int global_sum = inner_product(n, x, y);
    double tfinal = MPI_Wtime() - t0;
    MPI_Allreduce(&tfinal, &t0, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (my_id == 0) printf("Inner Product Time : %e\n", t0);

    MPI_Finalize();
    return 0;
}
