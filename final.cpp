#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <vector>

using namespace std;

// ===== Stable sigmoid =====
#pragma acc routine seq
double sigmoid(double z) {
    if (z > 35) return 1.0;
    if (z < -35) return 0.0;
    return 1.0 / (1.0 + exp(-z));
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ===== DATA =====
    int n = 0, m = 0;

    double* X = nullptr;
    double* y = nullptr;

    // ===== ROOT READS FILE =====
    if (rank == 0) {

        ifstream file("/mnt/c/Users/ASUS/.spyder-py3/weather_3M_75_25.csv");

        if (!file.is_open()) {
            cout << "ERROR: File not found\n";
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        string line;
        getline(file, line);

        vector<double> tempX;
        vector<double> tempY;

        while (getline(file, line)) {
            stringstream ss(line);
            string val;
            vector<double> row;

            while (getline(ss, val, ',')) {
                row.push_back(stod(val));
            }

            if (row.size() < 2) continue;

            tempY.push_back(row.back());
            row.pop_back();

            if (m == 0) m = row.size();

            for (double v : row) tempX.push_back(v);
        }

        file.close();

        n = tempY.size();

        X = new double[n * m];
        y = new double[n];

        for (int i = 0; i < n * m; i++) X[i] = tempX[i];
        for (int i = 0; i < n; i++) y[i] = tempY[i];

        cout << "Rows: " << n << " Features: " << m << endl;
    }

    // ===== BROADCAST SIZE =====
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ===== DISTRIBUTION =====
    int base = n / size;
    int rem  = n % size;

    int local_n = base + (rank < rem ? 1 : 0);

    int* counts = new int[size];
    int* displs = new int[size];

    for (int i = 0; i < size; i++) {
        counts[i] = base + (i < rem ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i-1] + counts[i-1];
    }

    // ===== LOCAL DATA =====
    double* local_X = new double[local_n * m];
    double* local_y = new double[local_n];

    int* counts_X = new int[size];
    int* displs_X = new int[size];

    for (int i = 0; i < size; i++) {
        counts_X[i] = counts[i] * m;
        displs_X[i] = displs[i] * m;
    }

    // ===== SCATTER =====
    MPI_Scatterv(X, counts_X, displs_X, MPI_DOUBLE,
                 local_X, local_n*m, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(y, counts, displs, MPI_DOUBLE,
                 local_y, local_n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    cout << "Rank " << rank << " has " << local_n << " rows\n";

    // ===== MODEL =====
    double* w = new double[m];
    double* dw = new double[m];

    for (int j = 0; j < m; j++) w[j] = 0.0;

    double b = 0.0;
    double lr = 0.05;
    int epochs = 1500;

    // ===== TIMER =====
    auto start = chrono::high_resolution_clock::now();

    // ===== GPU REGION =====
    #pragma acc data copyin(local_X[0:local_n*m], local_y[0:local_n]) \
                     copy(w[0:m]) create(dw[0:m])
    {
        for (int e = 0; e < epochs; e++) {

            // reset gradients
            #pragma acc parallel loop
            for (int j = 0; j < m; j++) dw[j] = 0.0;

            double db = 0.0;

            // compute gradients (NO ATOMIC!)
            #pragma acc parallel loop reduction(+:db)
            for (int i = 0; i < local_n; i++) {

                double z = b;

                for (int j = 0; j < m; j++)
                    z += w[j] * local_X[i*m + j];

                double pred = sigmoid(z);
                double err  = pred - local_y[i];

                // sequential accumulation per thread
                for (int j = 0; j < m; j++)
                    dw[j] += err * local_X[i*m + j];

                db += err;
            }

            // ===== MPI REDUCE =====
            MPI_Allreduce(MPI_IN_PLACE, dw, m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &db, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            // ===== UPDATE =====
            #pragma acc parallel loop
            for (int j = 0; j < m; j++)
                w[j] -= lr * dw[j] / n;

            b -= lr * db / n;

            if (rank == 0 && e % 50 == 0)
                cout << "Epoch " << e << endl;
        }
    }

    // ===== TIMER END =====
    auto end = chrono::high_resolution_clock::now();
    double time = chrono::duration<double>(end - start).count();

    if (rank == 0)
        cout << "\nExecution Time: " << time << " sec\n";

    // ===== CLEANUP =====
    delete[] local_X;
    delete[] local_y;
    delete[] w;
    delete[] dw;
    delete[] counts;
    delete[] displs;
    delete[] counts_X;
    delete[] displs_X;

    if (rank == 0) {
        delete[] X;
        delete[] y;
    }

    MPI_Finalize();
    return 0;
}
