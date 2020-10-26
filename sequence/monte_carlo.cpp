
#include "monte_carlo.hpp"


double linear_func(double x){
    return 2*x - 4;
}

template<typename return_type, typename arg_type>
double monte_carlo_seq(double A, double B, long long N, return_type(*f)(arg_type)){
    double min_val, max_val, area, x, y, result, points_in = 0;

    //random generator
    std::random_device rd;
    std::mt19937 gen{rd()};

    min_val = 0;//function calls in future
    max_val = 2;

    std::uniform_real_distribution<> dis_y(min_val, max_val);
    std::uniform_real_distribution<> dis_x(A, B);

    area = (B - A) * (max_val - min_val);

    for(long long i = 0; i < N; ++i){
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

//Test repeat 100 times and print average time of execution
void time_test(){
    std::cout << std::setprecision(5);
    std::chrono::duration<double> total;
    std::chrono::duration<double> diff;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    std::cout << "Testing sequence Monte Carlo..." << std::endl;
    for(int i = 1; i <= 100; ++i){
        std::cout << "\r" << i << "%  ";
        std::cout << std::flush;
        start = std::chrono::high_resolution_clock::now();
        monte_carlo_seq<double, double>(0, 2, 1000000, linear_func);
        end = std::chrono::high_resolution_clock::now();
        diff = end - start;
        total += diff;
    }

    std::cout << std::endl;
    std::cout << "Sequence Monte Carlo average time: " << total.count()/100 << std::endl;
}
