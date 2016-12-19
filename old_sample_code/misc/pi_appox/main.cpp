
/*

TODO:
    try these http://en.wikipedia.org/wiki/Approximations_of_%CF%80
        http://en.wikipedia.org/wiki/List_of_formulae_involving_%CF%80
    get rid of all the ints and use doubles since we might hit overflow on some systems
    finish writing readme adn better usage function

*/

#include <cmath>
#include <string>
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>

#define QUIT exit(1);

#define INT(x) ((int)(x))
#define DOUBLE(x) ((double)(x))
#define BYTE(x) ((unsigned char)(x))

#define SQUARE(x) ((x)*(x))
#define MOD(a,b) ( ((((int)a)%((int)b))+((int)b))%((int)b) )
#define MIN(a, b) ((a)<(b)?(a):(b))
#define MAX(a, b) ((a)>(b)?(a):(b))
#define RAND_FLOAT (((double) rand()) / (double)RAND_MAX)

#define DEBUG BUILD_DEBUG
#define ASSERT(x, msg) if (DEBUG && !(x)) { fprintf(stderr, "Assertion failed in %s(%d): %s\n", __FILE__, __LINE__, msg); assert(false); exit(1); }

#define TEST(x) cout << (#x) << ": " << x << endl;
#define PRINT(x) cout << x << endl;
#define DIVIDER cout << "======================================================================================================================================" << endl;
#define NEWLINE cout << endl;

using std::string;
using std::to_string;
using std::cout;
using std::endl;
using std::vector;

#define TIME_ELAPSED ((std::chrono::duration_cast<std::chrono::nanoseconds>(end_time-start_time).count()) / (pow(10.0,9.0)))

int double_factorial(int x){ // iterative for speed
    ASSERT(x>=0, "input to double_factorial must be >= 0");
    ASSERT(MOD(x,2)==1, "input to double_factorial must be odd");
    
    if (x==0) { return 1; }
    
    int ans = 0;
    
    for (unsigned int i=1; i<=x; i+=2) {
        ans *= i;
    }
    
    return ans;
}

int factorial(int x){ // iterative for speed
    ASSERT(x>=0, "input to factorial must be >= 0");
    
    if (x==0) { return 1; }
    
    int ans = 0;
    
    for (unsigned int i=1; i<=x; i++) {
        ans *= i;
    }
    
    return ans;
}

void monte_carlo_area_approx(double const &time_limit, double const &delta_value, int &num_iterations, double &approx_value, double &time_passed) {
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    
    int inside_count = 0;
    num_iterations = 0;
    start_time = std::chrono::high_resolution_clock::now();
    while(TIME_ELAPSED < time_limit || inside_count == 0) {
        double x = RAND_FLOAT;
        double y = RAND_FLOAT;
        double dist_squared = SQUARE(x)+SQUARE(y);
        if (dist_squared <= 1) {
            inside_count++;
        }
        num_iterations++;
        end_time = std::chrono::high_resolution_clock::now();
    }
    approx_value = 4.0*DOUBLE(inside_count)/DOUBLE(num_iterations);
    time_passed = TIME_ELAPSED;
}

void monte_carlo_volume_approx(double const &time_limit, double const &delta_value, int &num_iterations, double &approx_value, double &time_passed) {
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    
    int inside_count = 0;
    num_iterations = 0;
    start_time = std::chrono::high_resolution_clock::now();
    while(TIME_ELAPSED < time_limit || inside_count == 0) {
        double x = RAND_FLOAT;
        double y = RAND_FLOAT;
        double z = RAND_FLOAT;
        double dist_squared = SQUARE(x)+SQUARE(y)+SQUARE(z);
        if (dist_squared <= 1) {
            inside_count++;
        }
        num_iterations++;
        end_time = std::chrono::high_resolution_clock::now();
    }
    approx_value = 6.0*DOUBLE(inside_count)/DOUBLE(num_iterations);
    time_passed = TIME_ELAPSED;
}

void dovetail_approx(double const &time_limit, double const &delta_value, int &num_iterations, double &approx_value, double &time_passed) {
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    
    num_iterations = 0;
    int precision_level = 1;
    double delta;
    int inside_count = 3; // account for (1,0) and (0,1) and (0,0)
    int total_count = 1; // account for (1,1) 
    start_time = std::chrono::high_resolution_clock::now();
    while(TIME_ELAPSED < time_limit) {
        num_iterations++;
        delta = 1.0 / INT(1 << precision_level);
        for (double y=0.0; y<=1.0 && TIME_ELAPSED < time_limit; y+=delta) {
            for (double x=0.0; x<=1.0 && TIME_ELAPSED < time_limit; x+=delta) {
                if ( INT(x*(1<<(precision_level-1))) != DOUBLE(x*(1<<(precision_level-1)))  || INT(y*(1<<(precision_level-1))) != DOUBLE(y*(1<<(precision_level-1))) ) {
                    double dist_squared = SQUARE(x)+SQUARE(y);
                    if (dist_squared <= 1) {
                        inside_count++;
                    }
                    total_count++;
                }
                end_time = std::chrono::high_resolution_clock::now();   
            }
            end_time = std::chrono::high_resolution_clock::now();
        }
        precision_level++;
        end_time = std::chrono::high_resolution_clock::now();   
    }
    approx_value = 4.0*DOUBLE(inside_count)/DOUBLE(total_count);
    time_passed = TIME_ELAPSED;
}

// Approximations involving integrals

void method_1_approx(double const &time_limit, double const &delta_value, int &num_iterations, double &approx_value, double &time_passed) {
    // Approximation Method 1
        // From Wikipedia: en.wikipedia.org/wiki/List_of_formulae_involving_π
            // http://upload.wikimedia.org/math/4/4/2/44236100d35b1230c03c3d2832f65a9a.png
            // It's just the integral over a half circle
            // we're going to implement here an approximation where we'll just do a quarter circle
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    
    num_iterations = 0;
    approx_value = 0;
    start_time = std::chrono::high_resolution_clock::now();
    for (double x=0.0; x<1.0; x+=delta_value) {
        num_iterations++;
        approx_value += sqrt(1-SQUARE(x));
    }
    approx_value *= delta_value*4;
    end_time = std::chrono::high_resolution_clock::now();
    time_passed = TIME_ELAPSED;
}

void method_2_approx(double const &time_limit, double const &delta_value, int &num_iterations, double &approx_value, double &time_passed) {
    // Approximation Method 2
        // From Wikipedia: en.wikipedia.org/wiki/List_of_formulae_involving_π
            // http://upload.wikimedia.org/math/2/4/b/24b15f9ef0640e56efb055b284fedcb3.png
            // This fucntion is symetric accross y axis, so we can just integrate from 0 to 1 and double
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    
    num_iterations = 0;
    approx_value = 0;
    start_time = std::chrono::high_resolution_clock::now();
    for (double x=0.0; x<1.0; x+=delta_value) {
        num_iterations++;
        approx_value += 1/sqrt(1-SQUARE(x));
    }
    approx_value *= delta_value*2;
    end_time = std::chrono::high_resolution_clock::now();
    
    time_passed = TIME_ELAPSED;
}

void method_3_approx(double const &time_limit, double const &delta_value, int &num_iterations, double &approx_value, double &time_passed) {
    // Approximation Method 3
        // From Wikipedia: en.wikipedia.org/wiki/List_of_formulae_involving_π
            // http://upload.wikimedia.org/math/2/8/7/28751b5a35bf5c7ce59cb65fd15c6843.png
            // sech x = 1/cosh(x)
            // we'll only go in the positive direction and then just double it since the func is symmetric across y-axis
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    
    num_iterations = 0;
    approx_value = 0;
    start_time = std::chrono::high_resolution_clock::now();
    for (double x=0.0; TIME_ELAPSED < time_limit; x+=delta_value) {
        num_iterations++;
        approx_value += 1/cosh(x);
        end_time = std::chrono::high_resolution_clock::now();
    }
    approx_value *= 2*delta_value;
    end_time = std::chrono::high_resolution_clock::now();
    
    time_passed = TIME_ELAPSED;
}

void method_4_approx(double const &time_limit, double const &delta_value, int &num_iterations, double &approx_value, double &time_passed) {
    // Approximation Method 4
        // From Wikipedia: en.wikipedia.org/wiki/List_of_formulae_involving_π
            // upload.wikimedia.org/math/a/b/2/ab29074e1551c2d6a76fe7676ab85532.png
            // we'll only go in the positive direction and then just double it since the func is symmetric across y-axis
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    
    num_iterations = 0;
    approx_value = 0;
    start_time = std::chrono::high_resolution_clock::now();
    for (double x=0.0; TIME_ELAPSED < time_limit; x+=delta_value) {
        num_iterations++;
        approx_value += delta_value/(1+SQUARE(x));
        end_time = std::chrono::high_resolution_clock::now();
    }
    approx_value *= 2;
    end_time = std::chrono::high_resolution_clock::now();
    
    time_passed = TIME_ELAPSED;
}

void method_5_approx(double const &time_limit, double const &delta_value, int &num_iterations, double &approx_value, double &time_passed) {
    // Approximation Method 5
        // http://en.wikipedia.org/wiki/Gaussian_integral
            // we'll only go in the positive direction and then just double it since the func is symmetric across y-axis
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    
    num_iterations = 0;
    approx_value = 0;
    start_time = std::chrono::high_resolution_clock::now();
    for (double x=0.0; TIME_ELAPSED < time_limit; x+=delta_value) {
        num_iterations++;
        approx_value += exp(-SQUARE(x));
        end_time = std::chrono::high_resolution_clock::now();
    }
    approx_value *= 2*delta_value;
    approx_value = SQUARE(approx_value);
    end_time = std::chrono::high_resolution_clock::now();
    
    time_passed = TIME_ELAPSED;
}

void method_6_approx(double const &time_limit, double const &delta_value, int &num_iterations, double &approx_value, double &time_passed) {
    // Approximation Method 6
        // From Wikipedia: en.wikipedia.org/wiki/List_of_formulae_involving_π
            // http://upload.wikimedia.org/math/1/3/6/136d6f87aa10e23f51387e73376e3be6.png
            // we'll only go in the positive direction and then just double it since the func is symmetric across y-axis
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    
    num_iterations = 0;
    approx_value = delta_value; // to account for x = 0
    start_time = std::chrono::high_resolution_clock::now();
    for (double x=delta_value; TIME_ELAPSED < time_limit; x+=delta_value) {
        num_iterations++;
        approx_value += sin(x)/x;
        end_time = std::chrono::high_resolution_clock::now();
    }
    approx_value *= delta_value*2;
    end_time = std::chrono::high_resolution_clock::now();
    
    time_passed = TIME_ELAPSED;
}

void method_7_approx(double const &time_limit, double const &delta_value, int &num_iterations, double &approx_value, double &time_passed) {
    // Approximation Method 7
        // From Wikipedia: en.wikipedia.org/wiki/List_of_formulae_involving_π
            // http://upload.wikimedia.org/math/e/1/f/e1fd9dd4654fe9f43d6f907bd62b0c26.png
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    
    num_iterations = 0;
    approx_value = 0;
    start_time = std::chrono::high_resolution_clock::now();
    for (double x=0.0; x<1.0; x+=delta_value) {
        num_iterations++;
        approx_value += SQUARE(SQUARE(x))*SQUARE(SQUARE(1-x))/(1+SQUARE(x));
    }
    approx_value *= delta_value;
    approx_value -= 22.0/7.0;
    approx_value *= -1;
    end_time = std::chrono::high_resolution_clock::now();
    time_passed = TIME_ELAPSED;
}

void method_8_approx(double const &time_limit, double const &delta_value, int &num_iterations, double &approx_value, double &time_passed) {
    // Approximation Method 8
        // From Wikipedia: en.wikipedia.org/wiki/List_of_formulae_involving_π
            // http://upload.wikimedia.org/math/7/c/c/7cc81feb1b63af9b759364e41aeb091b.png
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    
    num_iterations = 0;
    approx_value = 0; 
    start_time = std::chrono::high_resolution_clock::now();
    for (double x=1; TIME_ELAPSED < time_limit; x+=1) {
        num_iterations++;
        approx_value += 1.0/(x*x);
        end_time = std::chrono::high_resolution_clock::now();
    }
    approx_value *= 6;
    approx_value = pow(approx_value, 0.5);
    end_time = std::chrono::high_resolution_clock::now();
    
    time_passed = TIME_ELAPSED;
}

void method_9_approx(double const &time_limit, double const &delta_value, int &num_iterations, double &approx_value, double &time_passed) {
    // Approximation Method 9
        // From Wikipedia: en.wikipedia.org/wiki/List_of_formulae_involving_π
            // http://upload.wikimedia.org/math/3/4/6/346d2b53812306e85db78030d8945c67.png
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    
    num_iterations = 0;
    approx_value = 0; 
    start_time = std::chrono::high_resolution_clock::now();
    for (double x=1; TIME_ELAPSED < time_limit; x+=1) {
        num_iterations++;
        approx_value += 1.0/(x*x*x*x);
        end_time = std::chrono::high_resolution_clock::now();
    }
    approx_value *= 90;
    approx_value = pow(approx_value, 0.25);
    end_time = std::chrono::high_resolution_clock::now();
    
    time_passed = TIME_ELAPSED;
}

void method_10_approx(double const &time_limit, double const &delta_value, int &num_iterations, double &approx_value, double &time_passed) {
    // Approximation Method 10
        // From Wikipedia: en.wikipedia.org/wiki/List_of_formulae_involving_π
            // http://upload.wikimedia.org/math/c/7/3/c73b92247c3b719e6b0fba1f28f497f3.png
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    
    num_iterations = 0;
    approx_value = 0; 
    start_time = std::chrono::high_resolution_clock::now();
    int sign = 1;
    double temp = 1;
    while(TIME_ELAPSED < time_limit) {
        num_iterations++;
        approx_value += (double)sign/temp;
        sign *= -1;
        temp += 2;
        end_time = std::chrono::high_resolution_clock::now();
    }
    approx_value *= 4;
    end_time = std::chrono::high_resolution_clock::now();
    
    time_passed = TIME_ELAPSED;
}

void method_11_approx(double const &time_limit, double const &delta_value, int &num_iterations, double &approx_value, double &time_passed) {
    // Approximation Method 11
        // From Wikipedia: en.wikipedia.org/wiki/List_of_formulae_involving_π
            // http://upload.wikimedia.org/math/b/f/9/bf97a3c5191aef8db2c0ea641fc7af65.png
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    
    num_iterations = 0;
    approx_value = 0; 
    double temp = 1;
    start_time = std::chrono::high_resolution_clock::now();
    while(TIME_ELAPSED < time_limit) {
        num_iterations++;
        approx_value += 1.0/(double)(temp*temp);
        temp += 2;
        end_time = std::chrono::high_resolution_clock::now();
    }
    approx_value = pow(approx_value*8,0.5);
    end_time = std::chrono::high_resolution_clock::now();
    
    time_passed = TIME_ELAPSED;
}

void method_12_approx(double const &time_limit, double const &delta_value, int &num_iterations, double &approx_value, double &time_passed) {
    // Approximation Method 12
        // From Wikipedia: en.wikipedia.org/wiki/List_of_formulae_involving_π
            // http://upload.wikimedia.org/math/8/e/0/8e0de730da027600ebce095ff332b5c7.png
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    
    num_iterations = 0;
    approx_value = 0; 
    start_time = std::chrono::high_resolution_clock::now();
    int sign = 1;
    double temp = 1;
    while (TIME_ELAPSED < time_limit) {
        num_iterations++;
        approx_value += (double)sign/(double)(temp*temp*temp);
        sign *= -1;
        temp += 2;
        end_time = std::chrono::high_resolution_clock::now();
    }
    approx_value = pow(approx_value*32,1.0/3.0);
    end_time = std::chrono::high_resolution_clock::now();
    
    time_passed = TIME_ELAPSED;
}

void usage() {
    fprintf(stderr, "usage: main <num_seconds_to_run_for_each_approximation> <delta_value>\n"
                    "\n"
                    "Tries many different strategies to approximate pi.\n"
                    "\n"
                    "The accuracy of each method is proportional to the amount of time spent approximating."
                    "The amount of time spent approximating, i.e. the number of iterations spent calculating, "
                    "for some methods can be prespecified. For other methods, it varies depending on a prespecified "
                    "precision value. These methods typically involve some sort of integration. Some methods may need "
                    "both a prespecified precision and run time."
                    "\n"
                    "num_seconds_to_run_for_each_approximation specifies how long approximation methods should run if "
                    "the runing time of the approximation can be prespecified."
                    "\n"
                    "delta_value species how precise the integrations will be."
           );
    exit(1);
}

int main(int argc, char* argv[]) {
    
    if (argc < 3) { 
        usage();
    }
    
    double time_limit = atof(argv[1]);
    double delta_value = atof(argv[2]);
    
    int num_iterations;
    double approx_value;
    double time_passed;
    
    cout.precision(100);
    
    PRINT("");
    PRINT("Starting Approximations.");
    PRINT("");
    
    DIVIDER;
    printf("%-80s %.50f\n", "cmath M_PI Value:", M_PI);
    DIVIDER;
    
    static const auto vector_of_funcs = vector<void (*)(const double&, const double &, int &, double &, double &)>
        {
//            monte_carlo_area_approx,
//            monte_carlo_volume_approx,
//            dovetail_approx,
//            method_1_approx,
//            method_2_approx,
//            method_3_approx,
//            method_4_approx,
//            method_5_approx,
//            method_6_approx,
//            method_7_approx,
//            method_8_approx,
//            method_9_approx,
            method_10_approx,
            method_11_approx,
            method_12_approx
        };
    
    static const auto vector_of_func_names = vector<string>
        {
//            string("Monte Carlo Area Approximation"), 
//            string("Monte Carlo Volume Approximation"), 
//            string("Dovetail Approximation"), 
//            string("Misc. Approximation Method 1 Approximation"), 
//            string("Misc. Approximation Method 2 Approximation"), 
//            string("Misc. Approximation Method 3 Approximation"), 
//            string("Misc. Approximation Method 4 Approximation"), 
//            string("Misc. Approximation Method 5 Approximation"), 
//            string("Misc. Approximation Method 6 Approximation"),
//            string("Misc. Approximation Method 7 Approximation"),
//            string("Misc. Approximation Method 8 Approximation"),
//            string("Misc. Approximation Method 9 Approximation"),
            string("Misc. Approximation Method 10 Approximation"),
            string("Misc. Approximation Method 11 Approximation"),
            string("Misc. Approximation Method 12 Approximation")
        };
    
    ASSERT(vector_of_func_names.size() == vector_of_funcs.size(), "number of funcs is not the same as the number of func names");
    
    for (unsigned i=0; i<vector_of_funcs.size(); i++) {
        (*(vector_of_funcs[i]))(time_limit, delta_value, num_iterations, approx_value, time_passed);
        NEWLINE;
        DIVIDER;
        printf("%-80s %.50f\n", (vector_of_func_names[i]+" Value:").c_str(), approx_value);
        printf("%-80s %.50f\n", (vector_of_func_names[i]+" Difference:").c_str(), fabs(approx_value-M_PI));
        printf("%-80s %d\n", "Iterations: ", num_iterations);
        printf("%-80s %f\n", "Run Time: ", time_passed);
        DIVIDER;
    }
    
    return 0;
}

