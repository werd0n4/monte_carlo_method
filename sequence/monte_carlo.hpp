
#pragma once
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>


double linear_func(double);

template<typename return_type, typename arg_type>
double monte_carlo_seq(double A, double B, long long N, return_type(*f)(arg_type));

void time_test();