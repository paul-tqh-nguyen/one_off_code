
/*

TODO:
    Add code to parse command line for arguments instead of just depending on them to be in the right place.
    move all final total initialization printing and such to the main.cpp file

*/

#include <chrono>
#include <cmath>
#include "patchmatch.h"
#include "../../util/array.h"
#include "../../util/pfm.h"

#define X_COORD 0
#define Y_COORD 1
#define D_COORD 2

#define RAND_FLOAT (DOUBLE(rand())/DOUBLE(RAND_MAX))
#define RAND_INT(lower, higher) (MOD(rand(),higher-lower)+lower)

using std::cout;
using std::endl;
using std::string;

void usage() {
    fprintf(stderr, "\n"
                    "usage: main <input_image_a>.png <input_image_b>.png <output_file>.pfm <patch_dim> <num_iterations> <random_search_size_exponent> <num_random_search_attempts>\n"
                    "\n"
                    "PatchMatch implementation. <output_file>.pfm will be a .pfm file containing the nearest \n"
                    "neighbor field. See http://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/index.php for further\n"
                    "explanation. <patch_dim> is an integer representing the patch height and width.\n"
                    "<random_search_size_exponent> determines the largest neighborhood size around a patch\n"
                    "in the nearest neighbor field to conduct the random search. When we are conducting the\n"
                    "random search around patch p in our nearest neighbor field, the largest neighborhood around p\n"
                    "we will search will be of size 2^<random_search_size_exponent>. <random_search_attempts> \n"
                    "defines how many neighbors we will compare to in each iteration of our random search.\n"
                    "\n"
           );
    exit(1);
}

int main(int argc, char* argv[]) {
    
    if (argc < 8) {
        usage();
    }
    
    std::srand( std::time(NULL) );
    
    static const char* A_name = argv[1];
    static const char* B_name = argv[2];
    static const char* output_name = argv[3];
    static const int patch_dim = atoi(argv[4]);
    static const int num_iterations = atoi(argv[5]);
    static const int random_search_size_exponent = atoi(argv[6]);
    static const int num_random_search_attempts = atoi(argv[7]);
    
    static const png::image< png::rgba_pixel > A_image(A_name); 
    static const png::image< png::rgba_pixel > B_image(B_name); 
    static const int A_height = A_image.get_height();
    static const int A_width = A_image.get_width();
    static const int B_height = B_image.get_height();
    static const int B_width = B_image.get_width();
    static const int Ann_height = A_height-patch_dim+1;
    static const int Ann_width = A_width-patch_dim+1;
    static const Array<byte> A(A_image);
    static const Array<byte> B(B_image);
    Array<unsigned long long> Ann(vector<int>{Ann_height, Ann_width, 3});
    
    NEWLINE;
    PRINT("Parameter Values");
    TEST(A_height);
    TEST(A_width);
    TEST(B_height);
    TEST(B_width);
    TEST(Ann_height);
    TEST(Ann_width);
    TEST(patch_dim);
    TEST(num_iterations);
    TEST(random_search_size_exponent);
    TEST(num_random_search_attempts);
    
    unsigned long long total_patch_distance;
    double mean_patch_distance;
    
    unsigned long long initial_total_patch_distance = 0;
    #pragma omp parallel for reduction(+:initial_total_patch_distance)
    for (int yy = 0; yy < Ann_height; ++yy) {
        for (int xx = 0; xx < Ann_width; ++xx) {
            initial_total_patch_distance += Ann(yy,xx,D_COORD);
        }
    }
    NEWLINE;
    cout << "Initial Total Patch Distance: " << initial_total_patch_distance << endl;
    cout << "Initial Mean Patch Distance:  " << DOUBLE(initial_total_patch_distance)/DOUBLE(Ann_height*Ann_width) << endl;
    fflush(stdout);
    
    patchmatch(A, B, Ann, A_height, A_width, B_height, B_width, Ann_height, Ann_width, patch_dim, num_iterations, random_search_size_exponent, num_random_search_attempts, total_patch_distance, mean_patch_distance);
    
    // Write output to .pfm file
    
    float *depth = new float[Ann_height*Ann_width*3];
    for (int y = 0; y < Ann_height; y++) {
        for (int x = 0; x < Ann_width; x++) {
            int i = (Ann_height-1-y)*Ann_width*3+x*3;
            
            depth[i] = FLOAT(Ann(y,x,X_COORD));
            depth[i+1] = FLOAT(Ann(y,x,Y_COORD));
            depth[i+2] = FLOAT(Ann(y,x,D_COORD));
        }
    }
    write_pfm_file3(output_name, depth, Ann_width, Ann_height);
    
    return 0;
}

