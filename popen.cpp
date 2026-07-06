#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <omp.h>   //  OpenMP

using namespace std;

// Stable sigmoid
double sigmoid(double z) {
    if (z > 35) return 1.0;
    if (z < -35) return 0.0;
    return 1.0 / (1.0 + exp(-z));
}

int main() {

    vector<vector<double>> X;
    vector<double> y;

    ifstream file("/mnt/c/Users/ASUS/.spyder-py3/weather_3M_75_25.csv");

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

    int n = X.size();
    int m = X[0].size();

    cout << "Rows: " << n << " Features: " << m << endl;

    // Class weights
    int count1 = 0;
    for (double v : y) if (v == 1) count1++;
    int count0 = n - count1;

    double w1 = (double)n / (2 * count1);
    double w0 = (double)n / (2 * count0);

    // Normalization
    for (int j = 0; j < m; j++) {

        double mean = 0.0;

        #pragma omp parallel for reduction(+:mean)
        for (int i = 0; i < n; i++)
            mean += X[i][j];

        mean /= n;

        double std_dev = 0.0;

        #pragma omp parallel for reduction(+:std_dev)
        for (int i = 0; i < n; i++)
            std_dev += (X[i][j] - mean) * (X[i][j] - mean);

        std_dev = sqrt(std_dev / n) + 1e-8;

        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            X[i][j] = (X[i][j] - mean) / std_dev;
    }

    cout << "Normalization done\n";

    vector<double> w(m, 0.0);
    double b = 0.0;

    double lr = 0.05;
    int epochs = 1000;
    double threshold = 0.4;

    auto start = chrono::high_resolution_clock::now();

    //  TRAINING (PARALLEL)
    for (int e = 0; e < epochs; e++) {

        vector<double> dw(m, 0.0);
        double db = 0.0;

        #pragma omp parallel
        {
            vector<double> local_dw(m, 0.0);
            double local_db = 0.0;

            #pragma omp for
            for (int i = 0; i < n; i++) {

                double z = b;
                for (int j = 0; j < m; j++)
                    z += w[j] * X[i][j];

                double pred = sigmoid(z);
                double err = (y[i] == 1 ? w1 : w0) * (pred - y[i]);

                for (int j = 0; j < m; j++)
                    local_dw[j] += err * X[i][j];

                local_db += err;
            }

            //  Combine threads
            #pragma omp critical
            {
                for (int j = 0; j < m; j++)
                    dw[j] += local_dw[j];

                db += local_db;
            }
        }

        for (int j = 0; j < m; j++)
            w[j] -= lr * dw[j] / n;

        b -= lr * db / n;

        if (e % 50 == 0)
            cout << "Epoch " << e << endl;
    }

    auto end = chrono::high_resolution_clock::now();
    double time = chrono::duration<double>(end - start).count();

    cout << "\nExecution Time: " << time << " sec\n";

    //  PARALLEL CONFUSION MATRIX
    int TP = 0, TN = 0, FP = 0, FN = 0;

    #pragma omp parallel for reduction(+:TP,TN,FP,FN)
    for (int i = 0; i < n; i++) {

        double z = b;
        for (int j = 0; j < m; j++)
            z += w[j] * X[i][j];

        double prob = sigmoid(z);
        int pred = (prob > threshold);
        int actual = y[i];

        if (pred == 1 && actual == 1) TP++;
        else if (pred == 0 && actual == 0) TN++;
        else if (pred == 1 && actual == 0) FP++;
        else FN++;
    }

    double accuracy = (double)(TP + TN) / n;
    double precision = (double)TP / (TP + FP);
    double recall = (double)TP / (TP + FN);
    double f1 = 2 * precision * recall / (precision + recall);

    cout << "\nAccuracy: " << accuracy * 100 << " %\n";
    cout << "Precision: " << precision << endl;
    cout << "Recall: " << recall << endl;
    cout << "F1 Score: " << f1 << endl;

    return 0;
}
