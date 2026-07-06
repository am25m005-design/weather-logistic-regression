#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>

using namespace std;

//  Stable Sigmoid
double sigmoid(double z) {
    if (z > 35) return 1.0;
    if (z < -35) return 0.0;
    return 1.0 / (1.0 + exp(-z));
}

int main() {
    vector<vector<double>> X;
    vector<double> y;

    ifstream file("/mnt/c/Users/ASUS/.spyder-py3/weather_3M_75_25.csv");

    if (!file.is_open()) {
        cout << "ERROR: File not found!" << endl;
        return 1;
    }

    string line;
    getline(file, line); 

    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<double> row;

        while (getline(ss, value, ',')) {
            row.push_back(stod(value));
        }

        if (row.size() < 2) continue;

        y.push_back(row.back());
        row.pop_back();
        X.push_back(row);
    }

    file.close();

    int n = X.size();
    int m = X[0].size();

    cout << "Rows loaded: " << n << endl;
    cout << "Features: " << m << endl;

    //  Class balance
    int count1 = 0;
    for (int i = 0; i < n; i++) {
        if (y[i] == 1) count1++;
    }

    int count0 = n - count1;

    cout << "Rain=1: " << count1 << " Rain=0: " << count0 << endl;

    double weight1 = (double)n / (2 * count1);
    double weight0 = (double)n / (2 * count0);

    cout << "Weight1: " << weight1 << " Weight0: " << weight0 << endl;

    //  Normalization
    for (int j = 0; j < m; j++) {

        double mean = 0.0, std_dev = 0.0;

        for (int i = 0; i < n; i++) mean += X[i][j];
        mean /= n;

        for (int i = 0; i < n; i++)
            std_dev += (X[i][j] - mean) * (X[i][j] - mean);

        std_dev = sqrt(std_dev / n) + 1e-8;

        for (int i = 0; i < n; i++)
            X[i][j] = (X[i][j] - mean) / std_dev;
    }

    cout << "Normalization done!\n";

    //  Model params
    vector<double> w(m, 0.0);
    vector<double> dw(m, 0.0);  //  reuse memory
    double b = 0.0;

    double lr = 0.05;
    int epochs = 1000;

    double threshold = 0.4;  // adjustable

    //  START TIMER
    auto start = chrono::high_resolution_clock::now();

    //  Training
    for (int e = 0; e < epochs; e++) {

        fill(dw.begin(), dw.end(), 0.0);
        double db = 0.0;

        for (int i = 0; i < n; i++) {

            double z = b;

            for (int j = 0; j < m; j++)
                z += w[j] * X[i][j];

            double pred = sigmoid(z);

            double error = (y[i] == 1 ? weight1 : weight0) * (pred - y[i]);

            for (int j = 0; j < m; j++)
                dw[j] += error * X[i][j];

            db += error;
        }

        for (int j = 0; j < m; j++)
            w[j] -= lr * dw[j] / n;

        b -= lr * db / n;

        if (e % 50 == 0)
            cout << "Epoch " << e << endl;
    }

    //  END TIMER
    auto end = chrono::high_resolution_clock::now();
    double time = chrono::duration<double>(end - start).count();

    cout << "\nExecution Time: " << time << " seconds\n";

    //  CONFUSION MATRIX
    int TP = 0, TN = 0, FP = 0, FN = 0;

    for (int i = 0; i < n; i++) {

        double z = b;
        for (int j = 0; j < m; j++)
            z += w[j] * X[i][j];

        double prob = sigmoid(z);
        int pred = (prob > threshold ? 1 : 0);
        int actual = (int)y[i];

        if (pred == 1 && actual == 1) TP++;
        else if (pred == 0 && actual == 0) TN++;
        else if (pred == 1 && actual == 0) FP++;
        else FN++;
    }

    //  METRICS
    double accuracy = (double)(TP + TN) / n;
    double precision = (TP + FP == 0) ? 0 : (double)TP / (TP + FP);
    double recall = (TP + FN == 0) ? 0 : (double)TP / (TP + FN);
    double f1 = (precision + recall == 0) ? 0 :
                2 * precision * recall / (precision + recall);

    // PRINT
    cout << "\nConfusion Matrix:\n";
    cout << "TP: " << TP << " FP: " << FP << endl;
    cout << "FN: " << FN << " TN: " << TN << endl;

    cout << "\nMetrics:\n";
    cout << "Accuracy : " << accuracy * 100 << " %\n";
    cout << "Precision: " << precision << endl;
    cout << "Recall   : " << recall << endl;
    cout << "F1 Score : " << f1 << endl;

    return 0;
}
