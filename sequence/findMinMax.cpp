#include <iostream>
#include <random>
#include <utility>
#include <numeric>

#include <iostream>
#include <chrono>
#include <iomanip>




typedef double(*FunctionCallback)(double);

namespace sequence {

std::pair<double, double> minMaxValue(int n, double a, double b, FunctionCallback func)
{
    
    std::random_device rd;
    std::mt19937 gen{rd()};

    std::uniform_real_distribution<> dis_x(a, b);

    double x, value;
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::lowest();
    for(int i =0; i<n; i++)
    {
        x = dis_x(gen);
        value = func(x);
        max = ( value > max ) ? value : max;
		min = ( value < min ) ? value : min;
    }

    return std::make_pair(min, max);
}

void timeTestMinMaxloSeq(int m, int n, double a, double b, FunctionCallback f){
    std::cout << std::setprecision(5);
    std::chrono::duration<double> total = std::chrono::duration<double>::zero();
    std::chrono::duration<double> diff;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    std::cout << "Testing sequence MinMax..." << std::endl;
    for(int i = 1; i <= m; ++i){
        start = std::chrono::high_resolution_clock::now();
        minMaxValue(n, a, b, f);
		end = std::chrono::high_resolution_clock::now();
        std::cout << "\r" << i * 100.0 / m << "%  ";
        std::cout << std::flush;
        diff = end - start;
        total += diff;
    }

    std::cout << std::endl;
    std::cout << "Sequence MinMax average time: " << total.count()/m << std::endl;
}

}

