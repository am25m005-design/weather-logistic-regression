#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <mpi.h>
#include <omp.h>

using namespace std;

// Stable sigmoid
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

    vector<vector<double>> X;
    vector<double> y;

    int n, m;

    // 🔹 ROOT READS DATA
    if (rank == 0) {
        ifstream file("/mnt/c/Users/ASUS/.spyder-py3/weather_3M_75_25.csv");

        if (!file.is_open()) {
            cout << "File not found!\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        string line;
        getline(file, line);

        while (getline(file, line)) {
            stringstream ss(line);
            string val;
            vector<double> row;

            while (getline(ss, val, ',')) row.push_back(stod(val));

            y.push_back(row.back());
            row.pop_back();
            X.push_back(row);
        }

        file.close();

        n = X.size();
        m = X[0].size();

        cout << "Total rows: " << n << " Features: " << m << endl;
    }

    // 🔹 BROADCAST SIZE
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 🔹 FLATTEN DATA (ROOT)
    vector<double> X_flat, y_flat;

    if (rank == 0) {
        X_flat.resize(n * m);
        y_flat.resize(n);

        for (int i = 0; i < n; i++) {
            y_flat[i] = y[i];
            for (int j = 0; j < m; j++)
                X_flat[i * m + j] = X[i][j];
        }
    }

    // 🔹 CREATE SCATTER INFO
    vector<int> counts(size), displs(size);

    int base = n / size;
    int rem = n % size;

    for (int i = 0; i < size; i++) {
        counts[i] = (i < rem) ? base + 1 : base;
        displs[i] = (i == 0) ? 0 : displs[i-1] + counts[i-1];
    }

    // 🔹 LOCAL DATA
    int local_n = counts[rank];
    vector<double> X_local(local_n * m);
    vector<double> y_local(local_n);

    // 🔹 SCATTER X
    vector<int> counts_X(size), displs_X(size);
    for (int i = 0; i < size; i++) {
        counts_X[i] = counts[i] * m;
        displs_X[i] = displs[i] * m;
    }

    MPI_Scatterv(X_flat.data(), counts_X.data(), displs_X.data(), MPI_DOUBLE,
                 X_local.data(), local_n * m, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // 🔹 SCATTER y
    MPI_Scatterv(y_flat.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 y_local.data(), local_n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    cout << "Rank " << rank << " has " << local_n << " rows\n";

    // 🔹 CLASS WEIGHTS (GLOBAL)
    int local_count1 = 0;
    for (double v : y_local) if (v == 1) local_count1++;

    int global_count1;
    MPI_Allreduce(&local_count1, &global_count1, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int global_count0 = n - global_count1;

    double w1 = (double)n / (2 * global_count1);
    double w0 = (double)n / (2 * global_count0);

    // 🔹 NORMALIZATION (GLOBAL MEAN/STD)
    for (int j = 0; j < m; j++) {

        double local_sum = 0.0;

        #pragma omp parallel for reduction(+:local_sum)
        for (int i = 0; i < local_n; i++)
            local_sum += X_local[i * m + j];

        double global_sum;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double mean = global_sum / n;

        double local_var = 0.0;

        #pragma omp parallel for reduction(+:local_var)
        for (int i = 0; i < local_n; i++) {
            double diff = X_local[i * m + j] - mean;
            local_var += diff * diff;
        }

        double global_var;
        MPI_Allreduce(&local_var, &global_var, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double std_dev = sqrt(global_var / n) + 1e-8;

        #pragma omp parallel for
        for (int i = 0; i < local_n; i++)
            X_local[i * m + j] = (X_local[i * m + j] - mean) / std_dev;
    }

    if (rank == 0) cout << "Normalization done\n";

    // 🔹 MODEL
    vector<double> w(m, 0.0);
    double b = 0.0;

    double lr = 0.05;
    int epochs = 1000;
    double threshold = 0.4;

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = chrono::high_resolution_clock::now();

    //  TRAINING
    for (int e = 0; e < epochs; e++) {

        vector<double> dw_local(m, 0.0), dw_global(m, 0.0);
        double db_local = 0.0, db_global = 0.0;

        #pragma omp parallel
        {
            vector<double> dw_private(m, 0.0);
            double db_private = 0.0;

            #pragma omp for
            for (int i = 0; i < local_n; i++) {

                double z = b;
                for (int j = 0; j < m; j++)
                    z += w[j] * X_local[i * m + j];

                double pred = sigmoid(z);
                double err = (y_local[i] == 1 ? w1 : w0) * (pred - y_local[i]);

                for (int j = 0; j < m; j++)
                    dw_private[j] += err * X_local[i * m + j];

                db_private += err;
            }

            #pragma omp critical
            {
                for (int j = 0; j < m; j++)
                    dw_local[j] += dw_private[j];
                db_local += db_private;
            }
        }

        MPI_Allreduce(dw_local.data(), dw_global.data(), m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&db_local, &db_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        for (int j = 0; j < m; j++)
            w[j] -= lr * dw_global[j] / n;

        b -= lr * db_global / n;

        if (rank == 0 && e % 50 == 0)
            cout << "Epoch " << e << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = chrono::high_resolution_clock::now();

    if (rank == 0) {
        double time = chrono::duration<double>(end - start).count();
        cout << "\nExecution Time: " << time << " sec\n";
    }

    // 🔹 EVALUATION (LOCAL → GLOBAL)
    int TP=0, TN=0, FP=0, FN=0;

    #pragma omp parallel for reduction(+:TP,TN,FP,FN)
    for (int i = 0; i < local_n; i++) {

        double z = b;
        for (int j = 0; j < m; j++)
            z += w[j] * X_local[i * m + j];

        double prob = sigmoid(z);
        int pred = (prob > threshold);
        int actual = y_local[i];

        if (pred == 1 && actual == 1) TP++;
        else if (pred == 0 && actual == 0) TN++;
        else if (pred == 1 && actual == 0) FP++;
        else FN++;
    }

    int gTP, gTN, gFP, gFN;

    MPI_Allreduce(&TP, &gTP, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&TN, &gTN, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&FP, &gFP, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&FN, &gFN, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        double acc = (double)(gTP + gTN) / n;
        double precision = (double)gTP / (gTP + gFP);
        double recall = (double)gTP / (gTP + gFN);
        double f1 = 2 * precision * recall / (precision + recall);

        cout << "\nAccuracy: " << acc * 100 << " %\n";
        cout << "Precision: " << precision << endl;
        cout << "Recall: " << recall << endl;
        cout << "F1 Score: " << f1 << endl;
    }

    MPI_Finalize();
    return 0;
}
