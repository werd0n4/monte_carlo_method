#pragma once
#include <random>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>

typedef double(*FunctionCallback)(double);

namespace sequence {

double monteCarlo(int N, double A, double B, double min, double max, FunctionCallback f){
    double area, x, randomValue, realValue, points_in = 0;

    //random generator
    std::random_device rd;
    std::mt19937 gen{rd()};


    std::uniform_real_distribution<> dis_y(min, max);
    std::uniform_real_distribution<> dis_x(A, B);

    area = (B - A) * (max - min);

    for(int i = 0; i < N; ++i){
        x = dis_x(gen);
        randomValue = dis_y(gen);
        realValue = f(x);

        if ((randomValue > 0) && (randomValue <= realValue)) {
            ++points_in;
        }
        else if ((randomValue < 0) && (randomValue >= realValue)) {
            --points_in;
        }
    }

    return area * points_in/N;
}

void timeTestMonteCarloSeq(int m, int n, double a, double b, double min, double max, FunctionCallback f){
    std::cout << std::setprecision(5);
    std::chrono::duration<double> total = std::chrono::duration<double>::zero();
    std::chrono::duration<double> diff;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;


    std::ofstream file;
    std::stringstream filename;
    filename << "monteSeq_" << m << '_' << n << ".txt";
    n = 1 << n;
    file.open(filename.str());
    if (file.good() == true)
    {

        std::cout << "Testing sequence Monte Carlo... for size: " << n << std::endl;
        for(int i = 1; i <= m; ++i){
            start = std::chrono::high_resolution_clock::now();
            monteCarlo(n, a, b, min, max, f);
	    	end = std::chrono::high_resolution_clock::now();
            std::cout << "\r" << i * 100.0 / m << "%  ";
            std::cout << std::flush;
            diff = end - start;
            file << diff.count() << std::endl;
            total += diff;
        }
    file.close();
    }

    std::cout << std::endl;
    std::cout << "Sequence Monte Carlo average time: " << total.count()/m << std::endl;
}

}