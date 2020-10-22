#pragma once
#include <random>


template<typename return_type, typename arg_type>
double monte_carlo_seq(double A, double B, int N, return_type(*f)(arg_type)){
    double min_val, max_val, area, x, y, result, points_in = 0;

    //random generator
    std::random_device rd;
    std::mt19937 gen{rd()};

    min_val = 0;//function calls in future
    max_val = 2;

    std::uniform_real_distribution<> dis_y(min_val, max_val);
    std::uniform_real_distribution<> dis_x(A, B);

    area = (B - A) * (max_val - min_val);

    for(int i = 0; i < N; ++i){
        x = dis_x(gen);
        y = dis_y(gen);
        result = f(x);

        if(result > 0 && y > 0){
            if(y < result){
                ++points_in;
            }
        }
        else if(result < 0 && y < 0){
            if(y > result){
                --points_in;
            }
        }
    }

    return area * points_in/N;
}

